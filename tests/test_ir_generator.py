"""
Tests for ir/generator.py — AST → LLVM IR lowering.

Tests are in two layers:
  1. End-to-end: lex → parse → analyze → generate_ir for each sample .c file,
     comparing against the ground-truth .ll file in tests/samples/.
  2. Unit: each AST construct tested in isolation with manually constructed ASTs.
"""

import os
import re
import glob
import pytest

from frontend.lexer import lex
from frontend.parser import parse
from frontend.semantic import analyze
from ir.generator import generate_ir
from frontend.ast import (
    Program, FunctionDef, Param, Block,
    VarDecl, Assignment, IfStmt, WhileStmt, ReturnStmt, ExprStmt,
    BinaryOp, NumberLiteral, Identifier, FunctionCall,
)

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(ir: str) -> str:
    """Strip trailing whitespace from each line and remove blank lines at end."""
    lines = [line.rstrip() for line in ir.splitlines()]
    # Remove trailing blank lines
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _pipeline(source: str) -> str:
    """Run the full lex → parse → analyze → generate_ir pipeline."""
    tokens = lex(source)
    ast = parse(tokens)
    ast = analyze(ast)
    return generate_ir(ast)


# ---------------------------------------------------------------------------
# End-to-end tests against ground-truth .ll files
# ---------------------------------------------------------------------------

def get_sample_pairs():
    """Return [(c_path, ll_path)] for all samples."""
    c_files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.c")))
    pairs = []
    for c_path in c_files:
        ll_path = c_path.replace(".c", ".ll")
        if os.path.exists(ll_path):
            pairs.append((c_path, ll_path))
    return pairs


@pytest.mark.parametrize("c_path,ll_path", get_sample_pairs())
def test_ir_matches_ground_truth(c_path, ll_path):
    """
    For each sample .c file, the generated IR should match the .ll ground truth
    after whitespace normalization.
    """
    with open(c_path, "r") as f:
        source = f.read()
    with open(ll_path, "r") as f:
        expected = f.read()

    generated = _pipeline(source)

    assert _normalize(generated) == _normalize(expected), (
        f"\n=== GENERATED ({os.path.basename(c_path)}) ===\n{generated}"
        f"\n=== EXPECTED ===\n{expected}"
    )


# ---------------------------------------------------------------------------
# Unit test: simple return (Example 1 from SPEC.md §3)
# ---------------------------------------------------------------------------

def test_simple_return():
    """
    Source:  int main() { return 42; }
    Expected IR:
        define i32 @main() {
        entry:
          ret i32 42
        }
    """
    source = "int main() { return 42; }"
    ir = _pipeline(source)
    assert "define i32 @main()" in ir
    assert "ret i32 42" in ir


# ---------------------------------------------------------------------------
# Unit test: variable declaration and addition (Example 2 from SPEC.md §3)
# ---------------------------------------------------------------------------

def test_var_decl_and_add():
    """
    Source:  int add(int a, int b) { int c = a + b; return c; }
    Checks:
      - alloca for a.addr, b.addr, c
      - store of parameters
      - load + add + store sequence
      - load + ret
    """
    source = "int add(int a, int b) { int c = a + b; return c; }"
    ir = _pipeline(source)

    assert "define i32 @add(i32 %a, i32 %b)" in ir
    assert "%a.addr = alloca i32" in ir
    assert "%b.addr = alloca i32" in ir
    assert "%c = alloca i32" in ir
    assert "store i32 %a, i32* %a.addr" in ir
    assert "store i32 %b, i32* %b.addr" in ir
    assert "= add i32" in ir
    assert "ret i32" in ir


# ---------------------------------------------------------------------------
# Unit test: if-else control flow (Example 3 from SPEC.md §3)
# ---------------------------------------------------------------------------

def test_if_else_control_flow():
    """
    Source:  int max(int x, int y) { if (x > y) { return x; } else { return y; } }
    Checks:
      - %retval alloca is hoisted to entry block
      - icmp sgt
      - br i1 ... if.then ... if.else
      - then block: load x, store retval, br return
      - else block: load y, store retval, br return
      - return block: load retval, ret
    """
    source = "int max(int x, int y) { if (x > y) { return x; } else { return y; } }"
    ir = _pipeline(source)

    assert "%retval = alloca i32" in ir
    assert "icmp sgt i32" in ir
    assert "if.then:" in ir
    assert "if.else:" in ir
    assert "return:" in ir
    assert "store i32" in ir
    assert "br label %return" in ir
    assert "ret i32" in ir


# ---------------------------------------------------------------------------
# Unit test: while loop (Sample 04_while.c)
# ---------------------------------------------------------------------------

def test_while_loop():
    """
    Source:  int count_to(int n) { int i = 0; while (i < n) { i = i + 1; } return i; }
    Checks:
      - alloca for n.addr and i
      - br label %while.cond
      - while.cond: icmp slt, br
      - while.body: load, add, store, br while.cond
      - while.end: load i, ret
    """
    source = (
        "int count_to(int n) {\n"
        "    int i = 0;\n"
        "    while (i < n) { i = i + 1; }\n"
        "    return i;\n"
        "}\n"
    )
    ir = _pipeline(source)

    assert "while.cond:" in ir
    assert "while.body:" in ir
    assert "while.end:" in ir
    assert "icmp slt i32" in ir
    assert "br label %while.cond" in ir


# ---------------------------------------------------------------------------
# Unit test: function call (Sample 05_func_call.c)
# ---------------------------------------------------------------------------

def test_function_call():
    """
    Source:  two functions, one calling the other.
    Checks:
      - both define blocks emitted
      - call instruction present
    """
    source = (
        "int double_it(int a) { return a * 2; }\n"
        "int main() { int res = double_it(5); return res; }\n"
    )
    ir = _pipeline(source)

    assert "define i32 @double_it(i32 %a)" in ir
    assert "define i32 @main()" in ir
    assert "call i32 @double_it(i32 5)" in ir
    assert "= mul i32" in ir


# ---------------------------------------------------------------------------
# Unit test: comparison operators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op,icmp_pred", [
    ("<",  "slt"),
    ("<=", "sle"),
    (">",  "sgt"),
    (">=", "sge"),
    ("==", "eq"),
    ("!=", "ne"),
])
def test_comparison_operator(op, icmp_pred):
    """Each relational/equality operator should produce the correct icmp predicate."""
    source = (
        f"int cmp(int a, int b) {{\n"
        f"    int r = 0;\n"
        f"    if (a {op} b) {{ r = 1; }}\n"
        f"    return r;\n"
        f"}}\n"
    )
    ir = _pipeline(source)
    assert f"icmp {icmp_pred} i32" in ir


# ---------------------------------------------------------------------------
# Unit test: multiple functions are separated by a blank line
# ---------------------------------------------------------------------------

def test_multiple_functions_separated():
    source = (
        "int foo() { return 1; }\n"
        "int bar() { return 2; }\n"
    )
    ir = _pipeline(source)
    assert "define i32 @foo()" in ir
    assert "define i32 @bar()" in ir
    # There should be an empty line between the two function blocks
    assert "\n\n" in ir


# ---------------------------------------------------------------------------
# Unit test: generate_ir raises TypeError on non-Program input
# ---------------------------------------------------------------------------

def test_generate_ir_type_error():
    with pytest.raises(TypeError):
        generate_ir(NumberLiteral(value=42))


# ---------------------------------------------------------------------------
# Unit test: arithmetic operations produce correct LLVM instructions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op,instr", [
    ("+", "add"),
    ("-", "sub"),
    ("*", "mul"),
    ("/", "sdiv"),
])
def test_arithmetic_ops(op, instr):
    source = f"int f(int a, int b) {{ return a {op} b; }}"
    ir = _pipeline(source)
    assert f"= {instr} i32" in ir


# ===========================================================================
# Regression tests: programs beyond the five samples must give valid IR
# ===========================================================================
#
# Each entry: (name, source, function, [(args, expected result), ...]).
# Expected results are what clang returns for the same program, with three
# exceptions where C leaves the result undefined and MiniC defines it
# (docs/SPEC.md §3, "Generator rules"):
#   * return_in_only_one_branch(-5), if_without_else_then_end(-5) and
#     shared_return_block_and_running_off_the_end(-1): control runs off the
#     end of the function, which returns 0 in MiniC.
#   * initializer_sees_outer_variable: in `int x = x + 5;` the right-hand `x`
#     is the outer variable in MiniC; in C it is the new, uninitialized one.
# The list is also used by tests/test_backend.py, which feeds every program
# to real LLVM.

PROGRAMS_BEYOND_SAMPLES = [
    ("if_on_plain_int",
     "int f(int a) { if (a) { return 1; } return 0; }",
     "f", [((0,), 0), ((5,), 1), ((-7,), 1)]),
    ("while_on_plain_int",
     "int f(int n) { int s = 0; while (n) { s = s + n; n = n - 1; } return s; }",
     "f", [((0,), 0), ((3,), 6), ((10,), 55)]),
    ("comparison_stored_in_variable",
     "int f(int a, int b) { int x = a < b; return x; }",
     "f", [((3, 4), 1), ((10, 3), 0), ((0, 0), 0)]),
    ("comparison_returned",
     "int f(int a, int b) { return a == b; }",
     "f", [((3, 4), 0), ((0, 0), 1)]),
    ("comparison_in_arithmetic",
     "int f(int a, int b) { return (a < b) + (a != b) * 10; }",
     "f", [((3, 4), 11), ((4, 3), 10), ((2, 2), 0)]),
    ("chained_comparison",          # (a < b) < c, as in C
     "int f(int a, int b, int c) { return a < b < c; }",
     "f", [((1, 2, 0), 0), ((1, 2, 5), 1), ((2, 1, 1), 1)]),
    ("comparison_as_call_argument",
     "int id(int v) { return v; } int f(int a) { return id(a > 2) + id(a > 100); }",
     "f", [((5,), 1), ((500,), 2), ((0,), 0)]),
    ("while_true_with_return_inside",
     "int f(int n) { int i = 0; while (1 == 1) { if (i * i >= n) { return i; } i = i + 1; } }",
     "f", [((0,), 0), ((10,), 4), ((100,), 10)]),
    ("while_literal_one",
     "int f(int n) { int i = 0; while (1) { if (i >= n) { return i; } i = i + 1; } }",
     "f", [((3,), 3), ((0,), 0)]),
    ("return_in_only_one_branch",   # the other path runs off the end → 0 (MiniC rule)
     "int f(int a) { if (a > 0) { return 1; } else { a = a + 1; } }",
     "f", [((5,), 1), ((-5,), 0)]),
    ("if_without_else_then_end",
     "int f(int a) { if (a > 0) { return 7; } }",
     "f", [((5,), 7), ((-5,), 0)]),
    ("shared_return_block_and_running_off_the_end",
     "int f(int a) { if (a > 0) { if (a > 5) { return 1; } else { return 2; } } }",
     "f", [((7,), 1), ((3,), 2), ((-1,), 0)]),
    ("empty_blocks",
     "int f(int a) { if (a) { } else { } while (a < 0) { a = a + 1; } if (a == 0) { } return a; }",
     "f", [((-3,), 0), ((4,), 4)]),
    ("statement_after_return",
     "int f(int a) { return a; a = a + 1; }",
     "f", [((5,), 5)]),
    ("return_after_return",
     "int f(int a) { return a; return a + 1; }",
     "f", [((5,), 5)]),
    ("code_after_if_else_both_return",
     "int f(int a) { if (a > 0) { return 1; } else { return 2; } return 3; }",
     "f", [((5,), 1), ((-5,), 2)]),
    ("dead_code_inside_branch",
     "int f(int a) { if (a > 0) { return 1; a = 9; } else { return 2; a = 8; } }",
     "f", [((5,), 1), ((-5,), 2)]),
    ("variable_named_add",
     "int f(int a, int b) { int add = a + b; return add; }",
     "f", [((3, 4), 7)]),
    ("variables_named_like_every_temporary",
     "int f(int a) { int add = a + 1; int sub = add - 2; int mul = sub * 3; "
     "int sdiv = mul / 2; int cmp = sdiv > 0; int conv = cmp == 1; "
     "int tobool = 0; if (conv) { tobool = 5; } return add + sub + mul + sdiv + cmp + conv + tobool; }",
     "f", [((5,), 6 + 4 + 12 + 6 + 1 + 1 + 5)]),
    ("variable_named_call",
     "int g(int v) { return v * 2; } int f(int a) { int call = g(a); return call + g(1); }",
     "f", [((5,), 12)]),
    ("param_named_mul",
     "int f(int mul, int b) { return mul * b; }",
     "f", [((3, 4), 12)]),
    ("variable_named_entry",
     "int f(int a) { int entry = a + 1; return entry; }",
     "f", [((5,), 6)]),
    ("param_named_entry",
     "int f(int entry) { return entry + 1; }",
     "f", [((5,), 6)]),
    ("variable_named_retval",
     "int f(int a) { int retval = 7; if (a > 0) { return retval; } else { return 0; } }",
     "f", [((5,), 7), ((-5,), 0)]),
    ("param_named_retval",
     "int f(int retval) { if (retval > 0) { return retval; } else { return 0; } }",
     "f", [((5,), 5), ((-5,), 0)]),
    ("shadowing_in_inner_scope",
     "int f(int a) { int x = 1; if (a > 0) { int x = 2; x = x + 1; } return x; }",
     "f", [((5,), 1), ((-5,), 1)]),
    ("inner_variable_shadows_parameter",
     "int f(int a) { if (a > 0) { int a = 100; a = a + 1; } return a; }",
     "f", [((5,), 5), ((-5,), -5)]),
    ("shadowing_in_loop_body",
     "int f(int n) { int x = 0; int i = 0; while (i < n) { int x = i * 10; i = i + 1; x = x + 1; } return x + i; }",
     "f", [((3,), 3)]),
    ("sibling_blocks_reuse_a_name",
     "int f(int a) { int r = 0; if (a > 0) { int t = 1; r = t; } else { int t = 2; r = t; } return r; }",
     "f", [((5,), 1), ((-5,), 2)]),
    ("initializer_sees_outer_variable",   # MiniC rule: init is evaluated before the name is declared
     "int f() { int x = 1; int y = 0; if (x > 0) { int x = x + 5; y = x; } return y * 10 + x; }",
     "f", [((), 61)]),
]

_PROGRAM_IDS = [p[0] for p in PROGRAMS_BEYOND_SAMPLES]
_TERMINATORS = ("ret ", "br ")


def _functions(ir: str):
    """Yield (signature_line, [(label, [instruction, ...]), ...]) per function."""
    current = None
    for line in ir.splitlines():
        if line.startswith("define "):
            current = (line, [])
        elif line.strip() == "}":
            yield current
            current = None
        elif current is not None and line.strip():
            if not line.startswith(" ") and line.rstrip().endswith(":"):
                current[1].append((line.rstrip()[:-1], []))
            else:
                current[1][-1][1].append(line.strip())


@pytest.mark.parametrize("name, source, fn, cases", PROGRAMS_BEYOND_SAMPLES, ids=_PROGRAM_IDS)
def test_every_block_has_exactly_one_terminator(name, source, fn, cases):
    for signature, blocks in _functions(_pipeline(source)):
        assert blocks, f"no blocks in {signature}"
        for label, instrs in blocks:
            assert instrs, f"block '{label}' is empty in {signature}"
            assert instrs[-1].startswith(_TERMINATORS), \
                f"block '{label}' does not end in a terminator: {instrs[-1]!r}"
            for instr in instrs[:-1]:
                assert not instr.startswith(_TERMINATORS), \
                    f"block '{label}' has a terminator before its end: {instr!r}"


@pytest.mark.parametrize("name, source, fn, cases", PROGRAMS_BEYOND_SAMPLES, ids=_PROGRAM_IDS)
def test_names_are_unique_within_a_function(name, source, fn, cases):
    """LLVM keeps parameters, values and labels of a function in one namespace."""
    for signature, blocks in _functions(_pipeline(source)):
        names = re.findall(r'i32 %([\w.]+)', signature.split("(", 1)[1])
        for label, instrs in blocks:
            names.append(label)
            for instr in instrs:
                m = re.match(r'%([\w.]+)\s*=', instr)
                if m:
                    names.append(m.group(1))
        duplicates = sorted({n for n in names if names.count(n) > 1})
        assert not duplicates, f"defined more than once in {signature}: {duplicates}"


@pytest.mark.parametrize("name, source, fn, cases", PROGRAMS_BEYOND_SAMPLES, ids=_PROGRAM_IDS)
def test_types_are_consistent(name, source, fn, cases):
    """A branch condition is always an i1; a comparison is never used as an i32."""
    ir = _pipeline(source)
    i1_values = set(re.findall(r'(%[\w.]+) = icmp ', ir))
    for line in ir.splitlines():
        m = re.match(r'\s+br i1 (%[\w.]+),', line)
        if m:
            assert m.group(1) in i1_values, f"branch on a non-i1 value: {line.strip()}"
            continue
        if " = zext i1 " in line:
            continue            # the one place an i1 may be an operand
        used = set(re.findall(r'%[\w.]+', line.split("=", 1)[-1]))
        assert not (used & i1_values), f"i1 value used as an integer: {line.strip()}"


@pytest.mark.parametrize("name, source, fn, cases", PROGRAMS_BEYOND_SAMPLES, ids=_PROGRAM_IDS)
def test_generated_code_computes_the_c_result(name, source, fn, cases):
    """Run the generated IR in the project's own interpreter."""
    from verifier.check import _parse_ir, _Interpreter
    module = _parse_ir(_pipeline(source))
    for args, expected in cases:
        assert _Interpreter(module).call(fn, list(args)) == expected, \
            f"{name}{args}"


def test_int_condition_is_compared_with_zero():
    ir = _pipeline("int f(int a) { if (a) { return 1; } return 0; }")
    assert "%tobool = icmp ne i32 %0, 0" in ir
    assert "br i1 %tobool, label %if.then, label %if.end" in ir


def test_comparison_condition_needs_no_conversion():
    ir = _pipeline("int f(int a, int b) { if (a < b) { return 1; } return 0; }")
    assert "br i1 %cmp, label %if.then, label %if.end" in ir
    assert "zext" not in ir and "tobool" not in ir


def test_comparison_value_is_zero_extended():
    ir = _pipeline("int f(int a, int b) { int x = a < b; return x; }")
    assert "%conv = zext i1 %cmp to i32" in ir
    assert "store i32 %conv, i32* %x" in ir


def test_statements_after_return_are_not_emitted():
    ir = _pipeline("int f(int a) { return a; a = a + 1; }")
    assert ir.count("ret i32") == 1
    assert "add i32" not in ir


def test_function_that_can_run_off_its_end_returns_zero():
    ir = _pipeline("int f(int a) { if (a > 0) { return 7; } }")
    assert "if.end:\n  ret i32 0\n" in ir


def test_shadowing_declarations_get_separate_slots():
    ir = _pipeline("int f(int a) { int x = 1; if (a > 0) { int x = 2; } return x; }")
    assert "  %x = alloca i32\n  %x.1 = alloca i32\n" in ir
    assert "store i32 1, i32* %x\n" in ir
    assert "store i32 2, i32* %x.1\n" in ir


def test_generated_names_step_around_user_names():
    ir = _pipeline("int f(int a, int b) { int add = a + b; return add; }")
    assert "  %add = alloca i32\n" in ir            # the variable keeps its name
    assert "  %add.1 = add i32 %0, %1\n" in ir      # the temporary moves aside
    assert "store i32 %add.1, i32* %add\n" in ir


def test_parameter_named_entry_does_not_clash_with_the_entry_block():
    ir = _pipeline("int f(int entry) { return entry + 1; }")
    assert "define i32 @f(i32 %entry.1) {\nentry:\n" in ir
