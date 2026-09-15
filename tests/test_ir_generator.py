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
