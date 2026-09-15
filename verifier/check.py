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
* Division by zero and step-limit overflows are treated as "runtime errors";
  if BOTH the original and optimised IRs crash identically, that counts as
  equivalent (the crash is a property of the input, not the optimisation).
"""

import re
from itertools import product
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

_MAX_STEPS   = 10_000   # guard against infinite loops in test inputs
_TEST_VALUES = [-1, 0, 1, 5, 42]  # concrete integers to test with
_MAX_TESTS   = 50       # cap total test cases for speed


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
    """Raised on division-by-zero or step-limit exceeded during interpretation."""


class _Frame:
    """Execution state for one function invocation."""
    def __init__(self):
        self.regs: Dict[str, int] = {}   # SSA register name (with %) → value
        self.mem:  Dict[str, int] = {}   # alloca slot name (with %) → value


class _Interpreter:
    """Pure-Python interpreter for our LLVM IR subset."""

    def __init__(self, module: _IRModule):
        self.module = module
        self._steps = 0

    def call(self, fn_name: str, args: List[int]) -> int:
        """Execute a function and return its i32 return value."""
        fn = self.module.functions.get(fn_name)
        if fn is None:
            raise _RuntimeError(f"Unknown function: @{fn_name}")

        frame = _Frame()
        # Pre-load parameters as SSA registers
        for param, val in zip(fn.params, args):
            frame.regs[f"%{param}"] = val

        return self._run(fn, frame)

    def _run(self, fn: _IRFunction, frame: _Frame) -> int:
        block_label = fn.entry
        while True:
            instrs = fn.blocks.get(block_label, [])
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
                raise _RuntimeError(f"Block '{block_label}' has no terminator")

    def _exec(self, line: str, frame: _Frame):
        """
        Execute one instruction string.
        Returns:
          None                 → instruction executed, continue
          ('ret', int_val)     → function should return this value
          ('br', label_str)    → jump to this block label
        """
        self._steps += 1
        if self._steps > _MAX_STEPS:
            raise _RuntimeError("Step limit exceeded (possible infinite loop)")

        # alloca
        m = _ALLOCA_RE.match(line)
        if m:
            frame.mem[m.group(1)] = 0
            return None

        # store
        m = _STORE_RE.match(line)
        if m:
            val  = self._resolve(m.group(1), frame)
            slot = m.group(2)
            frame.mem[slot] = val
            return None

        # load
        m = _LOAD_RE.match(line)
        if m:
            dst  = m.group(1)
            slot = m.group(2)
            frame.regs[dst] = frame.mem.get(slot, 0)
            return None

        # binary arithmetic
        m = _BINOP_RE.match(line)
        if m:
            dst, op, lhs_s, rhs_s = m.groups()
            lhs = self._resolve(lhs_s, frame)
            rhs = self._resolve(rhs_s, frame)
            frame.regs[dst] = self._arith(op, lhs, rhs)
            return None

        # icmp
        m = _ICMP_RE.match(line)
        if m:
            dst, pred, lhs_s, rhs_s = m.groups()
            lhs = self._resolve(lhs_s, frame)
            rhs = self._resolve(rhs_s, frame)
            frame.regs[dst] = self._icmp(pred, lhs, rhs)
            return None

        # conditional branch
        m = _BR_COND_RE.match(line)
        if m:
            cond_reg, true_lbl, false_lbl = m.groups()
            cond_val = self._resolve(cond_reg, frame)
            return ('br', true_lbl if cond_val else false_lbl)

        # unconditional branch
        m = _BR_UNCOND_RE.match(line)
        if m:
            return ('br', m.group(1))

        # ret
        m = _RET_RE.match(line)
        if m:
            return ('ret', self._resolve(m.group(1), frame))

        # call
        m = _CALL_RE.match(line)
        if m:
            dst, callee_name, args_str = m.groups()
            arg_vals = [self._resolve(v, frame)
                        for v in _CALL_ARG_RE.findall(args_str)]
            frame.regs[dst] = self.call(callee_name, arg_vals)
            return None

        # Anything else (blank, comment, opening/closing brace) — ignore
        return None

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve(operand: str, frame: _Frame) -> int:
        """Resolve an operand to an integer: either an SSA name or a literal."""
        operand = operand.strip()
        if operand.startswith('%'):
            # SSA register or alloca slot — check both
            if operand in frame.regs:
                return frame.regs[operand]
            if operand in frame.mem:
                return frame.mem[operand]
            raise _RuntimeError(f"Use of undefined register: {operand}")
        try:
            return int(operand)
        except ValueError:
            raise _RuntimeError(f"Cannot resolve operand: {operand!r}")

    @staticmethod
    def _arith(op: str, lhs: int, rhs: int) -> int:
        if op == 'add':  return lhs + rhs
        if op == 'sub':  return lhs - rhs
        if op == 'mul':  return lhs * rhs
        if op == 'sdiv':
            if rhs == 0:
                raise _RuntimeError("Division by zero")
            return int(lhs / rhs)  # truncate toward zero (C semantics)
        raise _RuntimeError(f"Unknown arithmetic op: {op}")

    @staticmethod
    def _icmp(pred: str, lhs: int, rhs: int) -> int:
        if pred == 'slt':  return int(lhs <  rhs)
        if pred == 'sle':  return int(lhs <= rhs)
        if pred == 'sgt':  return int(lhs >  rhs)
        if pred == 'sge':  return int(lhs >= rhs)
        if pred == 'eq':   return int(lhs == rhs)
        if pred == 'ne':   return int(lhs != rhs)
        raise _RuntimeError(f"Unknown icmp predicate: {pred}")


# ---------------------------------------------------------------------------
# Test input generation
# ---------------------------------------------------------------------------

def _generate_inputs(n_params: int) -> List[Tuple[int, ...]]:
    """
    Generate concrete integer input tuples for a function with n_params parameters.
    Uses the Cartesian product of _TEST_VALUES, capped at _MAX_TESTS.
    """
    if n_params == 0:
        return [()]
    combos = list(product(_TEST_VALUES, repeat=n_params))
    return combos[:_MAX_TESTS]


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
    integer inputs and returns True only if they produce identical outputs on
    every tested input combination.

    A return value of False means the optimization changed observable behaviour
    and must be rejected; the fallback pipeline will use the safe default pass
    order instead.

    Args:
        original_ir (str):   The unoptimized LLVM IR (output of generate_ir).
        optimized_ir (str):  The candidate optimized LLVM IR.

    Returns:
        bool: True if behaviourally equivalent on all tested inputs.
    """
    try:
        orig_module = _parse_ir(original_ir)
        opt_module  = _parse_ir(optimized_ir)
    except Exception:
        return False

    # Check that both modules export the same set of function names
    if set(orig_module.functions.keys()) != set(opt_module.functions.keys()):
        return False

    for fn_name, orig_fn in orig_module.functions.items():
        opt_fn = opt_module.functions.get(fn_name)
        if opt_fn is None:
            return False

        n_params = len(orig_fn.params)
        test_inputs = _generate_inputs(n_params)

        for args in test_inputs:
            # Run original
            orig_interp = _Interpreter(orig_module)
            try:
                orig_result = orig_interp.call(fn_name, list(args))
            except _RuntimeError:
                orig_result = None  # runtime error — treat as "crashed"

            # Run optimised
            opt_interp = _Interpreter(opt_module)
            try:
                opt_result = opt_interp.call(fn_name, list(args))
            except _RuntimeError:
                opt_result = None

            if orig_result != opt_result:
                return False  # outputs differ → not equivalent

    return True
