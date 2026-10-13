"""
Verifier for the MiniC compiler.

Two public functions:

  verify_ir(ir: str) -> bool
      Structural well-formedness check.  Returns True if the IR has the
      expected shape (at least one define, entry block, ret in each function).
      This matches the original interface in docs/SPEC.md §4.

  verify_equivalence(original_ir: str, optimized_ir: str) -> bool
      Differential testing.  Interprets both IRs on a set of concrete integer
      inputs and returns True only if they produce identical outputs on every
      input combination.  A mismatch means the optimization changed behaviour
      and must be rejected.

Design notes
------------
* Pure Python — no llvmlite, no subprocess, no external tools.
* The interpreter only needs to handle the LLVM IR subset that our generator
  (ir/generator.py) emits.  It is NOT a general LLVM IR interpreter.
* Differential testing is not a formal proof; it catches real bugs fast.
* Arithmetic is 32-bit two's complement, like LLVM's i32: results wrap, and
  `sdiv` truncates toward zero.
* A run of one function on one input ends in one of three ways:
    - a value            the function returned
    - a trap             division by zero, or INT_MIN / -1
    - inconclusive       the step limit or call-depth limit was hit first
  Value and trap are compared between the two IRs.  An inconclusive run of
  the ORIGINAL proves nothing, so that input is skipped; it is never counted
  as "both behave the same".  If a function has no conclusive input at all,
  equivalence is NOT established and the verifier answers False.
* IR the interpreter cannot execute (unknown instruction, undefined
  register, block without terminator) is never treated as equivalent.
"""

import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# IR parsing helpers
# ---------------------------------------------------------------------------

_DEFINE_RE   = re.compile(r'^define i32 @([\w]+)\((.*?)\)\s*\{')
_PARAM_RE    = re.compile(r'i32 %([\w.]+)')
_LABEL_RE    = re.compile(r'^([\w.]+):\s*(?:;.*)?$')

_ALLOCA_RE   = re.compile(r'^\s+(%[\w.]+)\s*=\s*alloca i32')
_STORE_RE    = re.compile(r'^\s+store i32 ([\w%.+\-]+),\s*i32\*\s*(%[\w.]+)')
_LOAD_RE     = re.compile(r'^\s+(%[\w.]+)\s*=\s*load i32,\s*i32\*\s*(%[\w.]+)')
_BINOP_RE    = re.compile(r'^\s+(%[\w.]+)\s*=\s*(add|sub|mul|sdiv)\s+i32\s+([\w%.\-]+),\s*([\w%.\-]+)')
_ICMP_RE     = re.compile(r'^\s+(%[\w.]+)\s*=\s*icmp\s+([\w]+)\s+i32\s+([\w%.\-]+),\s*([\w%.\-]+)')
_BR_COND_RE  = re.compile(r'^\s+br i1 (%[\w.]+),\s*label %([\w.]+),\s*label %([\w.]+)')
_BR_UNCOND_RE= re.compile(r'^\s+br label %([\w.]+)')
_RET_RE      = re.compile(r'^\s+ret i32\s+([\w%.\-]+)')
_CALL_RE     = re.compile(r'^\s+(%[\w.]+)\s*=\s*call i32 @([\w]+)\((.*?)\)')
_CALL_ARG_RE = re.compile(r'i32 ([\w%.\-]+)')

_MAX_STEPS      = 100_000  # per run; beyond this the run is "inconclusive"
_MAX_CALL_DEPTH = 100      # nested IR calls per run; beyond this "inconclusive"
_TEST_VALUES    = [-7, -1, 0, 1, 2, 5, 42]  # concrete integers to test with
_MAX_TESTS      = 50       # cap total test cases per function for speed

_INT32_MIN = -(2 ** 31)


def _wrap_i32(value: int) -> int:
    """Reduce a Python int to the signed 32-bit value LLVM's i32 would hold."""
    return ((value + 2 ** 31) % 2 ** 32) - 2 ** 31


# ---------------------------------------------------------------------------
# IR structure
# ---------------------------------------------------------------------------

class _IRFunction:
    """Parsed representation of a single LLVM IR function."""
    def __init__(self, name: str, params: List[str],
                 blocks: Dict[str, List[str]], entry: str):
        self.name   = name
        self.params = params          # list of bare parameter names (no %)
        self.blocks = blocks          # label → list of instruction strings
        self.entry  = entry           # label of the entry block


class _IRModule:
    """Parsed representation of a full IR file (collection of functions)."""
    def __init__(self, functions: Dict[str, _IRFunction]):
        self.functions = functions    # name → _IRFunction


def _parse_ir(ir_text: str) -> _IRModule:
    """Parse an LLVM IR text string into an _IRModule."""
    functions: Dict[str, _IRFunction] = {}
    lines = ir_text.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        m = _DEFINE_RE.match(line)
        if m:
            fn_name  = m.group(1)
            param_str = m.group(2)
            params   = _PARAM_RE.findall(param_str)

            # Collect function body until matching '}'
            blocks: Dict[str, List[str]] = {}
            current_label: Optional[str] = None
            current_instrs: List[str] = []
            entry_label: Optional[str] = None

            i += 1
            while i < len(lines):
                body_line = lines[i]
                i += 1

                # End of function
                if body_line.strip() == '}':
                    if current_label is not None:
                        blocks[current_label] = current_instrs
                    break

                # Strip inline comments (but keep the instruction)
                instr_line = re.sub(r'\s*;.*$', '', body_line)

                # Label line?
                lm = _LABEL_RE.match(instr_line)
                if lm:
                    if current_label is not None:
                        blocks[current_label] = current_instrs
                    current_label = lm.group(1)
                    current_instrs = []
                    if entry_label is None:
                        entry_label = current_label
                else:
                    if current_label is not None and instr_line.strip():
                        current_instrs.append(instr_line)

            if entry_label and blocks:
                functions[fn_name] = _IRFunction(
                    name=fn_name,
                    params=params,
                    blocks=blocks,
                    entry=entry_label,
                )
        else:
            i += 1

    return _IRModule(functions)


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

class _RuntimeError(Exception):
    """Base class for anything that stops a run before it returns a value."""


class _Trap(_RuntimeError):
    """The program itself faulted: division by zero or INT_MIN / -1."""


class _Inconclusive(_RuntimeError):
    """The step limit or call-depth limit was reached; the result is unknown."""


class _Malformed(_RuntimeError):
    """The IR cannot be executed (unknown instruction, undefined value, ...)."""


class _Frame:
    """Execution state for one function invocation."""
    def __init__(self):
        self.regs: Dict[str, int] = {}   # SSA register name (with %) → value
        self.mem:  Dict[str, int] = {}   # alloca slot name (with %) → value


class _Interpreter:
    """Pure-Python interpreter for our LLVM IR subset."""

    def __init__(self, module: _IRModule,
                 max_steps: Optional[int] = None,
                 max_call_depth: Optional[int] = None):
        self.module = module
        self._steps = 0
        self._depth = 0
        self._max_steps = _MAX_STEPS if max_steps is None else max_steps
        self._max_depth = (_MAX_CALL_DEPTH if max_call_depth is None
                           else max_call_depth)
        self._decoded: Dict[str, tuple] = {}   # instruction text → decoded form

    def call(self, fn_name: str, args: List[int]) -> int:
        """Execute a function and return its i32 return value."""
        fn = self.module.functions.get(fn_name)
        if fn is None:
            raise _Malformed(f"Unknown function: @{fn_name}")
        if len(args) != len(fn.params):
            raise _Malformed(
                f"@{fn_name} expects {len(fn.params)} argument(s), got {len(args)}")

        self._depth += 1
        if self._depth > self._max_depth:
            raise _Inconclusive("Call depth limit exceeded (possible infinite recursion)")
        try:
            frame = _Frame()
            # Pre-load parameters as SSA registers
            for param, val in zip(fn.params, args):
                frame.regs[f"%{param}"] = _wrap_i32(val)
            return self._run(fn, frame)
        finally:
            self._depth -= 1

    def _run(self, fn: _IRFunction, frame: _Frame) -> int:
        block_label = fn.entry
        while True:
            instrs = fn.blocks.get(block_label)
            if instrs is None:
                raise _Malformed(f"Branch to unknown block '{block_label}'")
            for instr in instrs:
                result = self._exec(instr, frame)
                if result is None:
                    continue
                kind = result[0]
                if kind == 'ret':
                    return result[1]
                elif kind == 'br':
                    block_label = result[1]
                    break   # restart block loop with new label
            else:
                # Block ended without a terminator — shouldn't happen in valid IR
                raise _Malformed(f"Block '{block_label}' has no terminator")

    # ── Decoding (done once per distinct instruction text) ──────────────────

    def _decode(self, line: str) -> tuple:
        cached = self._decoded.get(line)
        if cached is None:
            cached = self._decode_uncached(line)
            self._decoded[line] = cached
        return cached

    @staticmethod
    def _decode_uncached(line: str) -> tuple:
        m = _ALLOCA_RE.match(line)
        if m:
            return ('alloca', m.group(1))
        m = _STORE_RE.match(line)
        if m:
            return ('store', m.group(1), m.group(2))
        m = _LOAD_RE.match(line)
        if m:
            return ('load', m.group(1), m.group(2))
        m = _BINOP_RE.match(line)
        if m:
            return ('binop',) + m.groups()
        m = _ICMP_RE.match(line)
        if m:
            return ('icmp',) + m.groups()
        m = _BR_COND_RE.match(line)
        if m:
            return ('condbr',) + m.groups()
        m = _BR_UNCOND_RE.match(line)
        if m:
            return ('br', m.group(1))
        m = _RET_RE.match(line)
        if m:
            return ('ret', m.group(1))
        m = _CALL_RE.match(line)
        if m:
            dst, callee, args_str = m.groups()
            return ('call', dst, callee, tuple(_CALL_ARG_RE.findall(args_str)))
        return ('unknown', line.strip())

    def _exec(self, line: str, frame: _Frame):
        """
        Execute one instruction string.
        Returns:
          None                 → instruction executed, continue
          ('ret', int_val)     → function should return this value
          ('br', label_str)    → jump to this block label
        """
        self._steps += 1
        if self._steps > self._max_steps:
            raise _Inconclusive("Step limit exceeded (possible infinite loop)")

        op = self._decode(line)
        kind = op[0]

        if kind == 'load':
            _, dst, slot = op
            if slot not in frame.mem:
                raise _Malformed(f"Load from unknown slot: {slot}")
            frame.regs[dst] = frame.mem[slot]
            return None

        if kind == 'store':
            _, val_s, slot = op
            if slot not in frame.mem:
                raise _Malformed(f"Store to unknown slot: {slot}")
            frame.mem[slot] = self._resolve(val_s, frame)
            return None

        if kind == 'binop':
            _, dst, opcode, lhs_s, rhs_s = op
            lhs = self._resolve(lhs_s, frame)
            rhs = self._resolve(rhs_s, frame)
            frame.regs[dst] = self._arith(opcode, lhs, rhs)
            return None

        if kind == 'icmp':
            _, dst, pred, lhs_s, rhs_s = op
            lhs = self._resolve(lhs_s, frame)
            rhs = self._resolve(rhs_s, frame)
            frame.regs[dst] = self._icmp(pred, lhs, rhs)
            return None

        if kind == 'condbr':
            _, cond_reg, true_lbl, false_lbl = op
            cond_val = self._resolve(cond_reg, frame)
            return ('br', true_lbl if cond_val else false_lbl)

        if kind == 'br':
            return ('br', op[1])

        if kind == 'ret':
            return ('ret', self._resolve(op[1], frame))

        if kind == 'call':
            _, dst, callee_name, arg_names = op
            arg_vals = [self._resolve(v, frame) for v in arg_names]
            frame.regs[dst] = self.call(callee_name, arg_vals)
            return None

        if kind == 'alloca':
            # Uninitialised memory reads as 0 in both IRs being compared.
            frame.mem[op[1]] = 0
            return None

        # Never skip what we do not understand: a silently ignored instruction
        # would let two different programs look the same.
        raise _Malformed(f"Unrecognised instruction: {op[1]!r}")

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve(operand: str, frame: _Frame) -> int:
        """Resolve an operand to an integer: either an SSA name or a literal."""
        operand = operand.strip()
        if operand.startswith('%'):
            if operand in frame.regs:
                return frame.regs[operand]
            raise _Malformed(f"Use of undefined register: {operand}")
        try:
            return _wrap_i32(int(operand))
        except ValueError:
            raise _Malformed(f"Cannot resolve operand: {operand!r}")

    @staticmethod
    def _arith(op: str, lhs: int, rhs: int) -> int:
        if op == 'add':  return _wrap_i32(lhs + rhs)
        if op == 'sub':  return _wrap_i32(lhs - rhs)
        if op == 'mul':  return _wrap_i32(lhs * rhs)
        if op == 'sdiv':
            if rhs == 0:
                raise _Trap("Division by zero")
            if lhs == _INT32_MIN and rhs == -1:
                raise _Trap("Signed division overflow (INT_MIN / -1)")
            # Truncate toward zero (C / LLVM semantics), in exact integer maths.
            quotient = abs(lhs) // abs(rhs)
            return -quotient if (lhs < 0) != (rhs < 0) else quotient
        raise _Malformed(f"Unknown arithmetic op: {op}")

    @staticmethod
    def _icmp(pred: str, lhs: int, rhs: int) -> int:
        if pred == 'slt':  return int(lhs <  rhs)
        if pred == 'sle':  return int(lhs <= rhs)
        if pred == 'sgt':  return int(lhs >  rhs)
        if pred == 'sge':  return int(lhs >= rhs)
        if pred == 'eq':   return int(lhs == rhs)
        if pred == 'ne':   return int(lhs != rhs)
        raise _Malformed(f"Unknown icmp predicate: {pred}")


# ---------------------------------------------------------------------------
# Test input generation
# ---------------------------------------------------------------------------

def _generate_inputs(n_params: int) -> List[Tuple[int, ...]]:
    """
    Generate concrete integer input tuples for a function with n_params parameters.

    Uses the full Cartesian product of _TEST_VALUES when it fits in _MAX_TESTS.
    Otherwise it takes an evenly spaced sample of the product, so that every
    parameter still sees every test value (taking just the first _MAX_TESTS
    tuples would leave the leading parameters stuck on one or two values).
    """
    if n_params == 0:
        return [()]
    base = len(_TEST_VALUES)
    total = base ** n_params
    step = 1 if total <= _MAX_TESTS else -(-total // _MAX_TESTS)   # ceil
    if step > 1 and step % base == 0:
        step += 1      # a multiple of base would freeze the last parameter
    inputs: List[Tuple[int, ...]] = []
    for index in range(0, total, step):
        digits = []
        for _ in range(n_params):
            index, digit = divmod(index, base)
            digits.append(_TEST_VALUES[digit])
        inputs.append(tuple(reversed(digits)))
    return inputs[:_MAX_TESTS]


# ---------------------------------------------------------------------------
# Running one function on one input
# ---------------------------------------------------------------------------

_VALUE, _TRAP, _INCONCLUSIVE, _MALFORMED = "value", "trap", "inconclusive", "malformed"


def _run_once(module: _IRModule, fn_name: str, args: Tuple[int, ...],
              max_steps: int, max_call_depth: int) -> Tuple[str, object]:
    """
    Interpret one call and classify how it ended.
    Returns (kind, detail): kind is one of the four constants above; detail is
    the returned integer for a value, otherwise a short message.
    """
    interp = _Interpreter(module, max_steps=max_steps, max_call_depth=max_call_depth)
    try:
        return _VALUE, interp.call(fn_name, list(args))
    except _Trap as e:
        return _TRAP, str(e)
    except (_Inconclusive, RecursionError) as e:
        return _INCONCLUSIVE, str(e) or "recursion limit"
    except _RuntimeError as e:
        return _MALFORMED, str(e)


@lru_cache(maxsize=8)
def _compare(original_ir: str, optimized_ir: str,
             max_steps: int, max_call_depth: int) -> Optional[str]:
    """
    Core of the differential verifier.

    Returns None if the two IRs were shown to behave the same on the test
    inputs, otherwise a one-line reason why equivalence was not established.
    Cached, so asking for the verdict and then for the reason costs one run.
    """
    try:
        orig_module = _parse_ir(original_ir)
        opt_module  = _parse_ir(optimized_ir)
    except Exception as e:
        return f"IR could not be parsed: {e}"

    if not orig_module.functions:
        return "Original IR contains no functions"

    # Check that both modules export the same set of function names
    if set(orig_module.functions.keys()) != set(opt_module.functions.keys()):
        return "Function sets differ between original and optimized IR"

    for fn_name, orig_fn in orig_module.functions.items():
        opt_fn = opt_module.functions[fn_name]
        if len(opt_fn.params) != len(orig_fn.params):
            return f"@{fn_name}: parameter count changed"

        conclusive = 0
        for args in _generate_inputs(len(orig_fn.params)):
            orig_kind, orig_val = _run_once(orig_module, fn_name, args,
                                            max_steps, max_call_depth)
            if orig_kind == _MALFORMED:
                return f"@{fn_name}{args}: original IR cannot be executed ({orig_val})"
            if orig_kind == _INCONCLUSIVE:
                continue        # this input tells us nothing about either IR

            opt_kind, opt_val = _run_once(opt_module, fn_name, args,
                                          max_steps, max_call_depth)
            if opt_kind == _MALFORMED:
                return f"@{fn_name}{args}: optimized IR cannot be executed ({opt_val})"
            if opt_kind != orig_kind:
                return (f"@{fn_name}{args}: original ended with {orig_kind}, "
                        f"optimized with {opt_kind}")
            if orig_kind == _VALUE and orig_val != opt_val:
                return (f"@{fn_name}{args}: original returned {orig_val}, "
                        f"optimized returned {opt_val}")
            conclusive += 1

        if conclusive == 0:
            return (f"@{fn_name}: inconclusive, no test input finished within "
                    f"{max_steps} steps / call depth {max_call_depth}")

    return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def verify_ir(ir: str) -> bool:
    """
    Structural well-formedness check on the given LLVM IR string.

    Checks:
    - At least one 'define i32 @...' function is present.
    - Each function has a parseable entry block.
    - Each function body contains at least one 'ret i32' instruction.

    Args:
        ir (str): The LLVM IR to verify.

    Returns:
        bool: True if structurally valid, False otherwise.
    """
    try:
        module = _parse_ir(ir)
    except Exception:
        return False

    if not module.functions:
        return False

    for fn in module.functions.values():
        if not fn.blocks:
            return False
        # At least one block must contain a ret instruction
        has_ret = any(
            _RET_RE.match(instr)
            for instrs in fn.blocks.values()
            for instr in instrs
        )
        if not has_ret:
            return False

    return True


def verify_equivalence(original_ir: str, optimized_ir: str) -> bool:
    """
    Differential testing verifier.

    Interprets both the original and optimized LLVM IR on a set of concrete
    integer inputs and returns True only if they produce identical outcomes on
    every input the original finishes on, and every function has at least one
    such input.

    A return value of False means equivalence was NOT established: either the
    outputs differ, or the IR could not be executed, or no test input finished
    within the limits.  The pipeline must then not use the optimized IR.  Use
    equivalence_failure_reason() to find out which.

    Args:
        original_ir (str):   The unoptimized LLVM IR (output of generate_ir).
        optimized_ir (str):  The candidate optimized LLVM IR.

    Returns:
        bool: True if behaviourally equivalent on all tested inputs.
    """
    return _compare(original_ir, optimized_ir, _MAX_STEPS, _MAX_CALL_DEPTH) is None


def equivalence_failure_reason(original_ir: str, optimized_ir: str) -> Optional[str]:
    """
    Why verify_equivalence() answers False for this pair, as one line of text,
    or None if the pair is equivalent.  Shares its work with
    verify_equivalence(), so calling both does not interpret the IR twice.
    """
    return _compare(original_ir, optimized_ir, _MAX_STEPS, _MAX_CALL_DEPTH)
