"""
Tests for optimizer/passes.py — optimization pass library.

Each pass is tested independently with a crafted before-IR string, and the
expected after-IR string is checked.  The combined optimize() is also tested.

Test structure per pass:
  - test_<pass>_basic:    a simple before/after showing the core transformation
  - test_<pass>_no_op:    IR with no applicable patterns is returned unchanged
  - Additional edge-case tests where relevant

All IR strings are normalized (stripped trailing whitespace, blank trailing
lines removed) before comparison so formatting differences don't cause false
failures.
"""

import pytest
from optimizer.passes import (
    constant_folding,
    dead_code_elimination,
    common_subexpression_elimination,
    peephole,
    optimize,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def norm(ir: str) -> str:
    """Strip trailing whitespace per line, drop trailing blank lines."""
    lines = [line.rstrip() for line in ir.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


# ===========================================================================
# Pass 1: Constant Folding
# ===========================================================================

FOLD_BEFORE = """\
define i32 @fold_test() {
entry:
  %add = add i32 3, 4
  %mul = mul i32 2, 6
  ret i32 %add
}
"""

def test_constant_folding_basic():
    """
    add i32 3, 4 → the result (7) should propagate to downstream uses.
    mul i32 2, 6 → the result (12) should propagate.
    After folding %add = 7, the `ret i32 %add` should become `ret i32 7`.
    """
    result = constant_folding(FOLD_BEFORE)
    assert "ret i32 7" in result, (
        f"Expected 'ret i32 7' after constant folding.\nGot:\n{result}"
    )


def test_constant_folding_mul():
    """mul i32 2, 6 → 12 should appear in downstream uses."""
    ir = """\
define i32 @f() {
entry:
  %r = mul i32 2, 6
  ret i32 %r
}
"""
    result = constant_folding(ir)
    assert "ret i32 12" in result


def test_constant_folding_sub():
    ir = """\
define i32 @f() {
entry:
  %r = sub i32 10, 3
  ret i32 %r
}
"""
    result = constant_folding(ir)
    assert "ret i32 7" in result


def test_constant_folding_sdiv():
    ir = """\
define i32 @f() {
entry:
  %r = sdiv i32 10, 2
  ret i32 %r
}
"""
    result = constant_folding(ir)
    assert "ret i32 5" in result


def test_constant_folding_no_op():
    """If operands are registers, constant folding should not change anything."""
    ir = """\
define i32 @f(i32 %a, i32 %b) {
entry:
  %a.addr = alloca i32
  %b.addr = alloca i32
  store i32 %a, i32* %a.addr
  store i32 %b, i32* %b.addr
  %0 = load i32, i32* %a.addr
  %1 = load i32, i32* %b.addr
  %add = add i32 %0, %1
  ret i32 %add
}
"""
    result = constant_folding(ir)
    # The add should still be there (both operands are registers)
    assert "add i32 %0, %1" in result or "add i32" in result


def test_constant_folding_division_by_zero_skipped():
    """Division by zero should not be folded (leaves instruction intact)."""
    ir = """\
define i32 @f() {
entry:
  %r = sdiv i32 10, 0
  ret i32 %r
}
"""
    result = constant_folding(ir)
    # The instruction must NOT be replaced — it should still be present
    assert "sdiv i32 10, 0" in result or "%r" in result


# ===========================================================================
# Pass 2: Dead Code Elimination
# ===========================================================================

DCE_BEFORE = """\
define i32 @dce_test(i32 %a) {
entry:
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  %dead = add i32 %0, 1
  ret i32 %0
}
"""

def test_dce_removes_unused_def():
    """
    %dead = add i32 %0, 1 is defined but never used → should be eliminated.
    """
    result = dead_code_elimination(DCE_BEFORE)
    assert "%dead" not in result, (
        f"Expected %dead to be removed by DCE.\nGot:\n{result}"
    )


def test_dce_keeps_used_def():
    """Values that are actually used must NOT be eliminated."""
    ir = """\
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  %result = add i32 %0, 5
  ret i32 %result
}
"""
    result = dead_code_elimination(ir)
    assert "%result" in result


def test_dce_removes_unreachable_block():
    """
    A basic block that follows an unconditional `ret` and has no
    predecessor should be removed.
    """
    ir = """\
define i32 @f() {
entry:
  ret i32 42
dead.block:
  ret i32 0
}
"""
    result = dead_code_elimination(ir)
    assert "dead.block" not in result, (
        f"Expected unreachable block 'dead.block' to be removed.\nGot:\n{result}"
    )


def test_dce_no_op_when_all_used():
    """If every defined value is used, DCE should change nothing meaningful."""
    ir = """\
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  ret i32 %0
}
"""
    result = dead_code_elimination(ir)
    # %0 should still be present
    assert "load i32" in result
    assert "ret i32 %0" in result


def test_dce_chain_removal():
    """
    If a dead value's operands are also only used by that dead value,
    the chain should be eliminated iteratively.
    """
    ir = """\
define i32 @f() {
entry:
  %a = add i32 1, 2
  %b = add i32 %a, 3
  ret i32 10
}
"""
    result = dead_code_elimination(ir)
    # Both %a and %b are dead (ret uses constant 10)
    assert "%a" not in result
    assert "%b" not in result


# ===========================================================================
# Pass 3: Common Subexpression Elimination (CSE)
# ===========================================================================

CSE_BEFORE = """\
define i32 @cse_test(i32 %a, i32 %b) {
entry:
  %a.addr = alloca i32
  %b.addr = alloca i32
  store i32 %a, i32* %a.addr
  store i32 %b, i32* %b.addr
  %0 = load i32, i32* %a.addr
  %1 = load i32, i32* %b.addr
  %x = add i32 %0, %1
  %y = add i32 %0, %1
  ret i32 %y
}
"""

def test_cse_removes_duplicate():
    """
    %x = add i32 %0, %1  and  %y = add i32 %0, %1 are identical.
    CSE should remove %y and rewrite uses of %y to %x.
    """
    result = common_subexpression_elimination(CSE_BEFORE)
    assert "%y = add i32" not in result, (
        f"Expected %y duplicate to be removed by CSE.\nGot:\n{result}"
    )
    # ret should now use %x
    assert "ret i32 %x" in result


def test_cse_different_operands_kept():
    """Instructions with different operands must NOT be merged."""
    ir = """\
define i32 @f(i32 %a, i32 %b, i32 %c) {
entry:
  %a.addr = alloca i32
  %b.addr = alloca i32
  %c.addr = alloca i32
  store i32 %a, i32* %a.addr
  store i32 %b, i32* %b.addr
  store i32 %c, i32* %c.addr
  %0 = load i32, i32* %a.addr
  %1 = load i32, i32* %b.addr
  %2 = load i32, i32* %c.addr
  %x = add i32 %0, %1
  %y = add i32 %1, %2
  ret i32 %y
}
"""
    result = common_subexpression_elimination(ir)
    assert "%x = add i32 %0, %1" in result
    assert "%y = add i32 %1, %2" in result


def test_cse_no_op_when_no_duplicates():
    """IR with no duplicate expressions is returned unchanged."""
    ir = """\
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  %r = add i32 %0, 5
  ret i32 %r
}
"""
    result = common_subexpression_elimination(ir)
    assert "%r = add i32 %0, 5" in result


def test_cse_triple_duplicate():
    """If the same expression appears three times, only the first is kept."""
    ir = """\
define i32 @f(i32 %a, i32 %b) {
entry:
  %a.addr = alloca i32
  %b.addr = alloca i32
  store i32 %a, i32* %a.addr
  store i32 %b, i32* %b.addr
  %0 = load i32, i32* %a.addr
  %1 = load i32, i32* %b.addr
  %x = add i32 %0, %1
  %y = add i32 %0, %1
  %z = add i32 %0, %1
  ret i32 %z
}
"""
    result = common_subexpression_elimination(ir)
    # Only %x = add ... should remain
    assert "%y = add i32" not in result
    assert "%z = add i32" not in result
    assert "ret i32 %x" in result


# ===========================================================================
# Pass 4: Peephole
# ===========================================================================

def test_peephole_add_zero():
    """X + 0 should be eliminated; uses of result replaced with X."""
    ir = """\
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  %r = add i32 %0, 0
  ret i32 %r
}
"""
    result = peephole(ir)
    assert "ret i32 %0" in result, (
        f"Expected 'ret i32 %0' after eliminating X+0.\nGot:\n{result}"
    )


def test_peephole_zero_add():
    """0 + X should also be eliminated."""
    ir = """\
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  %r = add i32 0, %0
  ret i32 %r
}
"""
    result = peephole(ir)
    assert "ret i32 %0" in result


def test_peephole_sub_zero():
    """X - 0 should be eliminated."""
    ir = """\
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  %r = sub i32 %0, 0
  ret i32 %r
}
"""
    result = peephole(ir)
    assert "ret i32 %0" in result


def test_peephole_mul_one():
    """X * 1 should be eliminated."""
    ir = """\
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  %r = mul i32 %0, 1
  ret i32 %r
}
"""
    result = peephole(ir)
    assert "ret i32 %0" in result


def test_peephole_mul_zero():
    """X * 0 should be replaced with 0."""
    ir = """\
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  %r = mul i32 %0, 0
  ret i32 %r
}
"""
    result = peephole(ir)
    assert "ret i32 0" in result


def test_peephole_mul_two_strength_reduce():
    """
    X * 2 should be strength-reduced to add i32 X, X.
    """
    ir = """\
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  %r = mul i32 %0, 2
  ret i32 %r
}
"""
    result = peephole(ir)
    assert "add i32 %0, %0" in result, (
        f"Expected strength reduction mul→add.\nGot:\n{result}"
    )


def test_peephole_no_op():
    """IR with no peephole patterns is not changed."""
    ir = """\
define i32 @f(i32 %a, i32 %b) {
entry:
  %a.addr = alloca i32
  %b.addr = alloca i32
  store i32 %a, i32* %a.addr
  store i32 %b, i32* %b.addr
  %0 = load i32, i32* %a.addr
  %1 = load i32, i32* %b.addr
  %r = add i32 %0, %1
  ret i32 %r
}
"""
    result = peephole(ir)
    assert "add i32 %0, %1" in result


# ===========================================================================
# Combined: optimize() applies all passes in order
# ===========================================================================

def test_optimize_combined():
    """
    IR with a constant expression that folds to a constant, leaving a dead
    load that DCE can remove, demonstrating pass interaction.
    """
    ir = """\
define i32 @f() {
entry:
  %r = add i32 5, 3
  ret i32 %r
}
"""
    result = optimize(ir)
    # After constant folding %r = 8, the ret should use 8
    assert "ret i32 8" in result


def test_optimize_interface():
    """optimize() must accept a str and return a str."""
    ir = "define i32 @main() {\nentry:\n  ret i32 0\n}\n"
    result = optimize(ir)
    assert isinstance(result, str)
    assert "ret i32" in result


def test_optimize_peephole_and_dce():
    """
    mul X, 1 → X (peephole), then the dead add can be removed (DCE).
    All passes together should clean this up.
    """
    ir = """\
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  %r = mul i32 %0, 1
  %dead = add i32 %0, 99
  ret i32 %r
}
"""
    result = optimize(ir)
    # %dead should be gone
    assert "%dead" not in result
    # ret should return %0 directly
    assert "ret i32 %0" in result


# ===========================================================================
# Importability: each pass is callable by name (interface check)
# ===========================================================================

def test_each_pass_is_callable():
    """Verify all four passes and optimize are importable and callable."""
    test_ir = "define i32 @main() {\nentry:\n  ret i32 0\n}\n"
    for fn in [constant_folding, dead_code_elimination,
               common_subexpression_elimination, peephole, optimize]:
        result = fn(test_ir)
        assert isinstance(result, str), f"{fn.__name__} did not return str"


# ===========================================================================
# Regression tests: arithmetic must match LLVM's i32 semantics
# ===========================================================================

def _ret_main(body: str) -> str:
    return f"define i32 @main() {{\nentry:\n{body}}}\n"


@pytest.mark.parametrize("lhs, rhs, expected", [
    (-7, 2, -3),     # truncates toward zero, not down to -4
    (7, -2, -3),
    (-7, -2, 3),
    (7, 2, 3),
    (-1, 2, 0),
])
def test_constant_folding_sdiv_truncates_toward_zero(lhs, rhs, expected):
    ir = _ret_main(f"  %q = sdiv i32 {lhs}, {rhs}\n  ret i32 %q\n")
    assert f"ret i32 {expected}\n" in constant_folding(ir)


def test_constant_folding_chain_with_negative_intermediate():
    """(0 - 7) / 2 is -3 in C and in LLVM."""
    ir = _ret_main(
        "  %sub = sub i32 0, 7\n"
        "  %sdiv = sdiv i32 %sub, 2\n"
        "  ret i32 %sdiv\n"
    )
    assert "ret i32 -3\n" in constant_folding(ir)


def test_constant_folding_wraps_to_32_bits():
    """65536 * 65536 overflows i32 and wraps to 0; folding must agree."""
    ir = _ret_main(
        "  %mul = mul i32 65536, 65536\n"
        "  %sdiv = sdiv i32 %mul, 2\n"
        "  ret i32 %sdiv\n"
    )
    assert "ret i32 0\n" in constant_folding(ir)


def test_constant_folding_add_wraps_at_int_max():
    ir = _ret_main("  %add = add i32 2147483647, 1\n  ret i32 %add\n")
    assert "ret i32 -2147483648\n" in constant_folding(ir)


def test_constant_folding_int_min_div_minus_one_skipped():
    """INT_MIN / -1 overflows; like division by zero it must not be folded."""
    ir = _ret_main("  %q = sdiv i32 -2147483648, -1\n  ret i32 %q\n")
    assert "ret i32 %q\n" in constant_folding(ir)


# ===========================================================================
# Regression tests: DCE must not change behaviour or break the module
# ===========================================================================

def test_dce_keeps_unused_call():
    """A call may loop or trap, so it stays even if its result is unused."""
    ir = (
        "define i32 @g(i32 %a) {\nentry:\n  ret i32 %a\n}\n\n"
        "define i32 @main() {\nentry:\n"
        "  %call = call i32 @g(i32 1)\n"
        "  ret i32 5\n}\n"
    )
    assert "%call = call i32 @g(i32 1)" in dead_code_elimination(ir)


def test_dce_keeps_unused_division_that_may_trap():
    ir = (
        "define i32 @f(i32 %a, i32 %b) {\nentry:\n"
        "  %q = sdiv i32 %a, %b\n"
        "  ret i32 1\n}\n"
    )
    assert "%q = sdiv i32 %a, %b" in dead_code_elimination(ir)


def test_dce_removes_unused_division_by_nonzero_constant():
    ir = (
        "define i32 @f(i32 %a) {\nentry:\n"
        "  %q = sdiv i32 %a, 2\n"
        "  ret i32 1\n}\n"
    )
    assert "sdiv" not in dead_code_elimination(ir)


def test_dce_unreachable_last_block_keeps_closing_brace():
    """
    The closing '}' of a function that is not last in the module must survive
    when that function's final block is removed as unreachable.
    """
    ir = (
        "define i32 @a() {\nentry:\n  ret i32 1\n\ndead:\n  ret i32 2\n}\n\n"
        "define i32 @b() {\nentry:\n  ret i32 3\n}\n"
    )
    expected = (
        "define i32 @a() {\nentry:\n  ret i32 1\n\n}\n\n"
        "define i32 @b() {\nentry:\n  ret i32 3\n}\n"
    )
    assert dead_code_elimination(ir) == expected


def test_passes_leave_text_before_first_function_alone():
    ir = "; module comment\n\ndefine i32 @main() {\nentry:\n  ret i32 0\n}\n"
    for fn in [constant_folding, dead_code_elimination,
               common_subexpression_elimination, peephole, optimize]:
        assert fn(ir) == ir, f"{fn.__name__} changed a module it had nothing to do in"


# ===========================================================================
# Regression tests: unnamed temporaries must stay numbered %0, %1, %2, ...
# ===========================================================================

import re

NUMBERING_BEFORE = """\
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32
  %r = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  %mul = mul i32 0, %0
  store i32 %mul, i32* %r
  %1 = load i32, i32* %r
  ret i32 %1
}
"""


def _numbered_defs(ir: str):
    return [int(n) for n in re.findall(r'^\s+%(\d+)\s*=', ir, re.MULTILINE)]


def test_dce_renumbers_temporaries_after_removal():
    """
    peephole turns `mul 0, %0` into the constant 0, which makes `%0` dead.
    Once DCE deletes `%0`, the old `%1` has to become `%0` or LLVM rejects
    the function ("instruction expected to be numbered '%0'").
    """
    out = dead_code_elimination(peephole(NUMBERING_BEFORE))
    assert "%0 = load i32, i32* %a.addr" not in out      # the dead load is gone
    assert _numbered_defs(out) == [0]
    assert "%0 = load i32, i32* %r" in out
    assert "ret i32 %0" in out


def test_renumbering_is_a_no_op_when_nothing_was_removed():
    assert dead_code_elimination(NUMBERING_BEFORE) == NUMBERING_BEFORE


def test_cse_renumbers_temporaries_after_removal():
    ir = (
        "define i32 @f(i32 %a) {\nentry:\n"
        "  %0 = add i32 %a, 1\n"
        "  %1 = add i32 %a, 1\n"
        "  %2 = mul i32 %0, %1\n"
        "  ret i32 %2\n}\n"
    )
    out = common_subexpression_elimination(ir)
    assert _numbered_defs(out) == [0, 1]
    assert "%1 = mul i32 %0, %0" in out
    assert "ret i32 %1" in out


# ---------------------------------------------------------------------------
# Output encoding: optimized IR must be plain ASCII
# ---------------------------------------------------------------------------
# main.py writes the IR to a file and the pipeline may print it.  A non-ASCII
# character in a comment (it used to be "→") breaks writing on systems whose
# default encoding is not UTF-8, such as cp1252 on Windows.

def _source_to_ir(source: str) -> str:
    from frontend.lexer import lex
    from frontend.parser import parse
    from frontend.semantic import analyze
    from ir.generator import generate_ir
    return generate_ir(analyze(parse(lex(source))))


def _ascii_test_programs():
    import os
    from tests.test_ir_generator import PROGRAMS_BEYOND_SAMPLES
    samples = os.path.join(os.path.dirname(__file__), "samples")
    programs = {
        # peephole rewrites both identities and annotates the line
        "keep": "int f(int a) { return a * 1 + 0; }",
        # constant folding annotates the folded line
        "fold": "int main() { int x = 2 + 3; return x * 4; }",
    }
    for name in sorted(os.listdir(samples)):
        if name.endswith(".c"):
            with open(os.path.join(samples, name), encoding="utf-8") as f:
                programs[name] = f.read()
    for name, source, _fn, _cases in PROGRAMS_BEYOND_SAMPLES:
        programs[name] = source
    return programs


_PASSES = {
    "constant_folding": constant_folding,
    "dead_code_elimination": dead_code_elimination,
    "common_subexpression_elimination": common_subexpression_elimination,
    "peephole": peephole,
    "optimize": optimize,
}


@pytest.mark.parametrize("pass_name", sorted(_PASSES))
def test_every_pass_output_is_ascii(pass_name):
    fn = _PASSES[pass_name]
    for name, source in _ascii_test_programs().items():
        out = fn(_source_to_ir(source))
        bad = sorted({ch for ch in out if ord(ch) > 127})
        assert not bad, f"{pass_name} on {name}: non-ASCII {bad!r}"


def test_annotating_passes_were_exercised():
    """The ASCII test above must cover the passes that write comments."""
    assert "; peephole" in peephole(_source_to_ir("int f(int a) { return a * 1 + 0; }"))
    assert "; folded" in constant_folding(_source_to_ir("int main() { return 2 + 3; }"))
