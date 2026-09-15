"""
Optimization pass library for the MiniC compiler.

Implements four independently-callable optimization passes that operate on
LLVM IR text strings (pure Python, no llvmlite dependency):

    constant_folding(ir: str) -> str
        Evaluates arithmetic instructions whose operands are both integer
        literals and replaces them with the computed constant.

    dead_code_elimination(ir: str) -> str
        Removes instructions whose results (%name) are never used in the
        function body, and prunes unreachable basic blocks (blocks that
        follow an unconditional terminator and have no other predecessor).

    common_subexpression_elimination(ir: str) -> str
        Within each basic block, if the same opcode+operands appear more
        than once, replaces later uses of the duplicated result with the
        first SSA name that computed it.

    peephole(ir: str) -> str
        Applies a set of small-window algebraic identities:
          - add i32 X, 0  →  X
          - sub i32 X, 0  →  X
          - mul i32 X, 1  →  X
          - mul i32 X, 0  →  0
          - mul i32 X, 2  →  add i32 X, X  (strength reduction)

The public `optimize(ir: str) -> str` entry point runs all four passes in
the order above and returns the final optimized IR.

Interface defined in SPEC.md §4.
"""

import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared IR parsing helpers
# ---------------------------------------------------------------------------

# Matches an LLVM basic-block label line, e.g. "entry:" or "if.then:"
_BLOCK_LABEL_RE = re.compile(r'^([A-Za-z_.][A-Za-z0-9_.]*):')

# Matches a binary arithmetic instruction:
#   %result = add i32 %a, 7
#   %result = mul i32 3, 4
_BINOP_RE = re.compile(
    r'^\s+(%\S+)\s*=\s*(add|sub|mul|sdiv)\s+i32\s+([%\w\-.]+),\s*([%\w\-.]+)'
)

# Matches any SSA definition: %name = ...
_DEF_RE = re.compile(r'^\s+(%[A-Za-z0-9_.]+)\s*=')

# Matches a reference to an SSA value (used to detect liveness)
_USE_RE = re.compile(r'%[A-Za-z0-9_.]+')


def _split_into_functions(ir: str) -> List[str]:
    """Split IR text into per-function chunks (each starting with 'define')."""
    chunks: List[str] = []
    current: List[str] = []
    for line in ir.splitlines(keepends=True):
        if line.startswith("define ") and current:
            chunks.append("".join(current))
            current = []
        current.append(line)
    if current:
        chunks.append("".join(current))
    return chunks


def _split_into_blocks(lines: List[str]) -> List[Tuple[Optional[str], List[str]]]:
    """
    Split function body lines into basic blocks.
    Returns list of (label, lines). The entry block has label 'entry'.
    Lines before the first label are attached to an implicit block with label None.
    """
    blocks: List[Tuple[Optional[str], List[str]]] = []
    current_label: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        m = _BLOCK_LABEL_RE.match(line)
        if m:
            # Flush current block only if it has content or a non-None label
            if current_lines or current_label is not None:
                blocks.append((current_label, current_lines))
            current_label = m.group(1)
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines or current_label is not None:
        blocks.append((current_label, current_lines))

    # Drop any leading empty-lines block with no label (artifact of splitting)
    while blocks and blocks[0][0] is None and not blocks[0][1]:
        blocks.pop(0)

    return blocks


def _blocks_to_lines(blocks: List[Tuple[Optional[str], List[str]]]) -> List[str]:
    """Reassemble blocks back into a flat line list."""
    out: List[str] = []
    for label, lines in blocks:
        if label is not None:
            out.append(f"{label}:")
        out.extend(lines)
    return out


# ---------------------------------------------------------------------------
# Pass 1: Constant Folding
# ---------------------------------------------------------------------------

_FOLD_OPS = {
    'add':  lambda a, b: a + b,
    'sub':  lambda a, b: a - b,
    'mul':  lambda a, b: a * b,
    'sdiv': lambda a, b: a // b if b != 0 else None,
}

_IMMEDIATE_RE = re.compile(r'^-?\d+$')


def _is_immediate(val: str) -> bool:
    return bool(_IMMEDIATE_RE.match(val))


def _fold_function(fn_text: str) -> str:
    """Apply constant folding within a single function body."""
    lines = fn_text.splitlines(keepends=True)

    # We track constants as we go: if %x was folded to a literal, record it.
    known: Dict[str, int] = {}

    result: List[str] = []
    for line in lines:
        m = _BINOP_RE.match(line)
        if m:
            dst, op, lhs, rhs = m.group(1), m.group(2), m.group(3), m.group(4)

            # Resolve lhs/rhs through known constants
            lhs_val = known.get(lhs, lhs)
            rhs_val = known.get(rhs, rhs)

            if _is_immediate(str(lhs_val)) and _is_immediate(str(rhs_val)):
                folded_fn = _FOLD_OPS.get(op)
                if folded_fn is not None:
                    computed = folded_fn(int(lhs_val), int(rhs_val))
                    if computed is not None:
                        # Record that dst is now a known constant.
                        known[dst] = computed
                        # Emit the instruction but replace the whole RHS with
                        # the computed value, so downstream uses see a constant.
                        # Actually, to properly propagate, we emit a copy-like
                        # comment and substitute dst in downstream lines.
                        # For textual IR: we simply note the value; downstream
                        # loads of dst as an operand will be substituted below.
                        # Elide the instruction entirely (dead after folding).
                        result.append(
                            line.rstrip('\n').rstrip('\r') +
                            f"  ; folded → {computed}\n"
                        )
                        continue
        # Substitute any known constants into this line's operands
        new_line = line
        for reg, val in known.items():
            # Replace register references but not partial matches (word boundary)
            new_line = re.sub(
                re.escape(reg) + r'(?=[,\s)\n\r]|$)',
                str(val),
                new_line
            )
        result.append(new_line)

    return "".join(result)


def constant_folding(ir: str) -> str:
    """
    Applies constant folding to the given LLVM IR string.

    Any binary arithmetic instruction whose both operands are integer
    literals is replaced with the computed result.  Downstream uses of the
    folded register are substituted with the constant value.

    Args:
        ir: LLVM IR text string.

    Returns:
        Optimized LLVM IR text string.
    """
    fns = _split_into_functions(ir)
    return "".join(_fold_function(fn) for fn in fns)


# ---------------------------------------------------------------------------
# Pass 2: Dead Code Elimination
# ---------------------------------------------------------------------------

def _dce_function(fn_text: str) -> str:
    """Remove dead instructions within a single function."""
    lines = fn_text.splitlines(keepends=True)

    # Separate header ("define …") and closing "}"
    if not lines:
        return fn_text
    header = lines[0]
    if lines and lines[-1].strip() == "}":
        footer = lines[-1]
        body_lines = lines[1:-1]
    else:
        footer = ""
        body_lines = lines[1:]

    # ── Phase A: remove unreachable blocks ──────────────────────────────────
    # A basic block is unreachable if it is not the entry block AND no
    # terminator in a preceding block names it as a branch target.
    blocks = _split_into_blocks(body_lines)
    body_lines = _remove_unreachable_blocks(blocks)

    # ── Phase B: remove unused SSA definitions ───────────────────────────────
    # Collect all definitions and all uses across the full body.
    defs: List[str] = []  # ordered list of defined %names
    for line in body_lines:
        m = _DEF_RE.match(line)
        if m:
            defs.append(m.group(1))

    # Count uses of each defined name (exclude the definition line itself)
    use_count: Dict[str, int] = {d: 0 for d in defs}
    for line in body_lines:
        # Find all %name references in this line
        refs = _USE_RE.findall(line)
        for ref in refs:
            if ref in use_count:
                # Don't count the LHS definition itself
                lhs_m = _DEF_RE.match(line)
                if lhs_m and lhs_m.group(1) == ref:
                    continue
                use_count[ref] += 1

    # An instruction is dead if its result is defined but never used AND
    # the instruction has no side effects (i.e., not store/call/br/ret).
    _SIDE_EFFECT_RE = re.compile(r'^\s+(store|call|br|ret|;)')

    changed = True
    while changed:
        changed = False
        new_body: List[str] = []
        for line in body_lines:
            m = _DEF_RE.match(line)
            if m:
                name = m.group(1)
                if (use_count.get(name, 1) == 0
                        and not _SIDE_EFFECT_RE.match(line)):
                    # Dead — omit and decrement use counts of its operands
                    refs = _USE_RE.findall(line)
                    for ref in refs:
                        if ref != name and ref in use_count:
                            use_count[ref] = max(0, use_count[ref] - 1)
                    changed = True
                    continue
            new_body.append(line)
        body_lines = new_body

    result_lines = [header] + body_lines + ([footer] if footer else [])
    return "".join(result_lines)


def _remove_unreachable_blocks(
    blocks: List[Tuple[Optional[str], List[str]]]
) -> List[str]:
    """
    Remove basic blocks that are not reachable from the entry block.
    Returns the reassembled body lines.
    """
    if not blocks:
        return []

    # Collect all branch targets across all blocks
    _BR_TARGET_RE = re.compile(r'label %([A-Za-z0-9_.]+)')

    def targets_of(block_lines: List[str]) -> List[str]:
        ts = []
        for line in block_lines:
            ts.extend(_BR_TARGET_RE.findall(line))
        return ts

    # Build reachability set via BFS from entry
    label_to_block = {}
    for label, blines in blocks:
        label_to_block[label] = blines

    reachable = set()
    # Start BFS from the first block (entry) — use its actual label
    entry_label = blocks[0][0] if blocks else None
    queue = [entry_label]
    while queue:
        lbl = queue.pop(0)
        if lbl in reachable:
            continue
        reachable.add(lbl)
        blines = label_to_block.get(lbl, [])
        for t in targets_of(blines):
            if t not in reachable:
                queue.append(t)

    filtered = [(lbl, blines) for lbl, blines in blocks if lbl in reachable]
    return _blocks_to_lines(filtered)


def dead_code_elimination(ir: str) -> str:
    """
    Applies dead code elimination to the given LLVM IR string.

    Removes:
    - Instructions whose SSA result is defined but never used (and have
      no side effects).
    - Unreachable basic blocks (blocks with no path from the entry block).

    Args:
        ir: LLVM IR text string.

    Returns:
        Optimized LLVM IR text string.
    """
    fns = _split_into_functions(ir)
    return "".join(_dce_function(fn) for fn in fns)


# ---------------------------------------------------------------------------
# Pass 3: Common Subexpression Elimination (CSE)
# ---------------------------------------------------------------------------

def _cse_function(fn_text: str) -> str:
    """Apply CSE within each basic block of a single function."""
    lines = fn_text.splitlines(keepends=True)
    if not lines:
        return fn_text

    header = lines[0]
    if lines and lines[-1].strip() == "}":
        footer = lines[-1]
        body_lines = lines[1:-1]
    else:
        footer = ""
        body_lines = lines[1:]

    blocks = _split_into_blocks(body_lines)
    new_blocks: List[Tuple[Optional[str], List[str]]] = []

    for label, blines in blocks:
        new_blocks.append((label, _cse_block(blines)))

    new_body = _blocks_to_lines(new_blocks)
    result_lines = [header] + new_body + ([footer] if footer else [])
    return "".join(result_lines)


# Matches arithmetic / icmp instructions for CSE
_CSE_INSTR_RE = re.compile(
    r'^\s+(%\S+)\s*=\s*(add|sub|mul|sdiv|icmp \w+)\s+i32\s+([%\w\-.]+),\s*([%\w\-.]+)'
)


def _cse_block(lines: List[str]) -> List[str]:
    """
    Within a single block, replace later instructions that duplicate an
    earlier one with a substitution of the earlier SSA name.
    """
    # expr_key → first SSA name that computed it
    seen: Dict[str, str] = {}
    # substitution map: %dup → %first
    subst: Dict[str, str] = {}

    result: List[str] = []
    for line in lines:
        # Apply existing substitutions to this line first
        new_line = _apply_subst(line, subst)
        m = _CSE_INSTR_RE.match(new_line)
        if m:
            dst, op, lhs, rhs = m.group(1), m.group(2), m.group(3), m.group(4)
            key = f"{op} i32 {lhs}, {rhs}"
            if key in seen:
                # This is a duplicate — record substitution, skip line
                subst[dst] = seen[key]
                continue
            else:
                seen[key] = dst
        result.append(new_line)

    return result


def _apply_subst(line: str, subst: Dict[str, str]) -> str:
    """Replace all %reg references in line according to substitution map."""
    for old, new in subst.items():
        line = re.sub(
            re.escape(old) + r'(?=[,\s)\n\r]|$)',
            new,
            line
        )
    return line


def common_subexpression_elimination(ir: str) -> str:
    """
    Applies common subexpression elimination (CSE) to the given LLVM IR.

    Within each basic block, if the same arithmetic or comparison instruction
    (same opcode and same operands) appears more than once, later occurrences
    are removed and all uses of their result are redirected to the first
    occurrence's SSA name.

    Args:
        ir: LLVM IR text string.

    Returns:
        Optimized LLVM IR text string.
    """
    fns = _split_into_functions(ir)
    return "".join(_cse_function(fn) for fn in fns)


# ---------------------------------------------------------------------------
# Pass 4: Peephole
# ---------------------------------------------------------------------------

# Pattern: %dst = add|sub i32 %X, 0  →  emit comment; replace dst with X
# Pattern: %dst = mul i32 %X, 1      →  replace dst with X
# Pattern: %dst = mul i32 %X, 0      →  replace dst with 0
# Pattern: %dst = mul i32 %X, 2      →  %dst = add i32 %X, %X
# All symmetric variants (0+X etc.) are also handled.

_PEEP_RE = re.compile(
    r'^(\s+)(%\S+)\s*=\s*(add|sub|mul|sdiv)\s+i32\s+([%\w\-.]+),\s*([%\w\-.]+)'
)


def _peephole_function(fn_text: str) -> str:
    """Apply peephole optimizations within a single function."""
    lines = fn_text.splitlines(keepends=True)
    subst: Dict[str, str] = {}  # register substitutions across whole function

    result: List[str] = []
    for line in lines:
        # Apply pending substitutions
        new_line = _apply_subst(line, subst)

        m = _PEEP_RE.match(new_line)
        if m:
            indent, dst, op, lhs, rhs = (
                m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            )
            replacement = _peephole_rule(op, lhs, rhs)
            if replacement is not None:
                if isinstance(replacement, str):
                    # dst is equivalent to a simpler value
                    subst[dst] = replacement
                    # Elide the original instruction
                    result.append(
                        new_line.rstrip('\n').rstrip('\r') +
                        f"  ; peephole → {replacement}\n"
                    )
                    continue
                elif isinstance(replacement, tuple):
                    # Emit a rewritten instruction
                    new_op, new_lhs, new_rhs = replacement
                    result.append(
                        f"{indent}{dst} = {new_op} i32 {new_lhs}, {new_rhs}\n"
                    )
                    continue

        result.append(new_line)

    return "".join(result)


def _peephole_rule(op: str, lhs: str, rhs: str):
    """
    Return:
      - a str if the instruction's result equals that operand/constant
      - a tuple (new_op, new_lhs, new_rhs) if the instruction should be
        rewritten
      - None if no rule applies
    """
    lhs_is_zero = (lhs == '0')
    rhs_is_zero = (rhs == '0')
    lhs_is_one  = (lhs == '1')
    rhs_is_one  = (rhs == '1')
    lhs_is_two  = (lhs == '2')
    rhs_is_two  = (rhs == '2')

    if op == 'add':
        if rhs_is_zero:
            return lhs          # X + 0 = X
        if lhs_is_zero:
            return rhs          # 0 + X = X

    elif op == 'sub':
        if rhs_is_zero:
            return lhs          # X - 0 = X

    elif op == 'mul':
        if rhs_is_one:
            return lhs          # X * 1 = X
        if lhs_is_one:
            return rhs          # 1 * X = X
        if rhs_is_zero or lhs_is_zero:
            return '0'          # X * 0 = 0
        if rhs_is_two:
            return ('add', lhs, lhs)   # X * 2 → X + X
        if lhs_is_two:
            return ('add', rhs, rhs)   # 2 * X → X + X

    elif op == 'sdiv':
        if rhs_is_one:
            return lhs          # X / 1 = X

    return None


def peephole(ir: str) -> str:
    """
    Applies peephole optimizations to the given LLVM IR string.

    Handles the following algebraic identities:
      - add i32 X, 0  →  X  (and 0 + X)
      - sub i32 X, 0  →  X
      - mul i32 X, 1  →  X  (and 1 * X)
      - mul i32 X, 0  →  0  (and 0 * X)
      - mul i32 X, 2  →  add i32 X, X  (strength reduction)
      - sdiv i32 X, 1 →  X

    Args:
        ir: LLVM IR text string.

    Returns:
        Optimized LLVM IR text string.
    """
    fns = _split_into_functions(ir)
    return "".join(_peephole_function(fn) for fn in fns)


# ---------------------------------------------------------------------------
# Public interface (SPEC.md §4)
# ---------------------------------------------------------------------------

def optimize(ir: str) -> str:
    """
    Applies optimization passes to the given LLVM IR string and returns
    the optimized IR.

    Passes applied in order:
      1. constant_folding
      2. dead_code_elimination
      3. common_subexpression_elimination
      4. peephole

    Each pass is also callable independently via the named functions above.

    Args:
        ir: The unoptimized LLVM IR text string.

    Returns:
        The optimized LLVM IR text string.
    """
    ir = constant_folding(ir)
    ir = dead_code_elimination(ir)
    ir = common_subexpression_elimination(ir)
    ir = peephole(ir)
    return ir
