"""
Unit tests for the MiniC verifier (verifier/check.py).

Tests cover:
- verify_ir: structural well-formedness checks
- verify_equivalence: differential testing on all 5 ground-truth sample programs
- verify_equivalence: known-equivalent and known-inequivalent IR pairs
- The pure-Python IR interpreter internals
"""

import os
import pytest

from verifier.check import verify_ir, verify_equivalence, _parse_ir, _Interpreter

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_sample(name: str) -> str:
    """Read a .ll sample file."""
    with open(os.path.join(SAMPLES_DIR, name)) as f:
        return f.read()


def pipeline_ir(c_filename: str) -> str:
    """Run the full frontend → IR pipeline on a sample .c file."""
    from frontend.lexer import lex
    from frontend.parser import parse
    from frontend.semantic import analyze
    from ir.generator import generate_ir
    with open(os.path.join(SAMPLES_DIR, c_filename)) as f:
        src = f.read()
    return generate_ir(analyze(parse(lex(src))))


# ---------------------------------------------------------------------------
# verify_ir — structural checks
# ---------------------------------------------------------------------------

class TestVerifyIr:
    def test_valid_simple_function(self):
        ir = "define i32 @main() {\nentry:\n  ret i32 42\n}\n"
        assert verify_ir(ir) is True

    def test_valid_with_params(self):
        ir = "define i32 @add(i32 %a, i32 %b) {\nentry:\n  ret i32 0\n}\n"
        assert verify_ir(ir) is True

    def test_empty_string_is_invalid(self):
        assert verify_ir("") is False

    def test_no_ret_is_invalid(self):
        ir = "define i32 @main() {\nentry:\n  %x = add i32 1, 2\n}\n"
        assert verify_ir(ir) is False

    def test_no_define_is_invalid(self):
        ir = "entry:\n  ret i32 0\n"
        assert verify_ir(ir) is False

    def test_multiple_functions_all_valid(self):
        ir = (
            "define i32 @f() {\nentry:\n  ret i32 1\n}\n\n"
            "define i32 @g() {\nentry:\n  ret i32 2\n}\n"
        )
        assert verify_ir(ir) is True

    def test_sample_golden_files_are_valid(self):
        for n in ['01_return', '02_add', '03_if_else', '04_while', '05_func_call']:
            ir = read_sample(f"{n}.ll")
            assert verify_ir(ir) is True, f"{n}.ll failed verify_ir"


# ---------------------------------------------------------------------------
# verify_equivalence — known-equivalent pairs
# ---------------------------------------------------------------------------

class TestVerifyEquivalenceEquivalent:
    def test_identical_ir_is_equivalent(self):
        ir = "define i32 @main() {\nentry:\n  ret i32 42\n}\n"
        assert verify_equivalence(ir, ir) is True

    def test_sample_01_return_golden(self):
        orig = pipeline_ir("01_return.c")
        assert verify_equivalence(orig, orig) is True

    def test_sample_02_add_golden(self):
        orig = pipeline_ir("02_add.c")
        assert verify_equivalence(orig, orig) is True

    def test_sample_03_if_else_golden(self):
        orig = pipeline_ir("03_if_else.c")
        assert verify_equivalence(orig, orig) is True

    def test_sample_04_while_golden(self):
        orig = pipeline_ir("04_while.c")
        assert verify_equivalence(orig, orig) is True

    def test_sample_05_func_call_golden(self):
        orig = pipeline_ir("05_func_call.c")
        assert verify_equivalence(orig, orig) is True

    def test_after_full_optimize_still_equivalent(self):
        """Optimizer output must always be equivalent to the original."""
        from optimizer.passes import optimize
        for n in ['01_return', '02_add', '03_if_else', '04_while', '05_func_call']:
            orig = pipeline_ir(f"{n}.c")
            opt  = optimize(orig)
            assert verify_equivalence(orig, opt) is True, \
                f"optimize({n}.c) produced non-equivalent IR"

    def test_after_individual_passes(self):
        """Each individual pass must produce equivalent IR."""
        from optimizer.passes import (
            constant_folding, dead_code_elimination,
            common_subexpression_elimination, peephole
        )
        orig = pipeline_ir("02_add.c")
        for pass_fn in [constant_folding, dead_code_elimination,
                        common_subexpression_elimination, peephole]:
            opt = pass_fn(orig)
            assert verify_equivalence(orig, opt) is True, \
                f"{pass_fn.__name__} produced non-equivalent IR for 02_add.c"


# ---------------------------------------------------------------------------
# verify_equivalence — known-inequivalent pairs
# ---------------------------------------------------------------------------

class TestVerifyEquivalenceInequivalent:
    def test_different_return_constant(self):
        orig = "define i32 @main() {\nentry:\n  ret i32 42\n}\n"
        broken = "define i32 @main() {\nentry:\n  ret i32 99\n}\n"
        assert verify_equivalence(orig, broken) is False

    def test_wrong_arithmetic(self):
        """
        Original: return a + b
        Broken: return a - b
        Should fail for inputs where a != b.
        """
        orig = (
            "define i32 @add(i32 %a, i32 %b) {\n"
            "entry:\n"
            "  %a.addr = alloca i32\n"
            "  %b.addr = alloca i32\n"
            "  store i32 %a, i32* %a.addr\n"
            "  store i32 %b, i32* %b.addr\n"
            "  %0 = load i32, i32* %a.addr\n"
            "  %1 = load i32, i32* %b.addr\n"
            "  %add = add i32 %0, %1\n"
            "  ret i32 %add\n"
            "}\n"
        )
        broken = orig.replace("add i32", "sub i32")
        assert verify_equivalence(orig, broken) is False

    def test_missing_function(self):
        """Optimized IR drops a function entirely — should fail."""
        orig = (
            "define i32 @f() {\nentry:\n  ret i32 1\n}\n\n"
            "define i32 @g() {\nentry:\n  ret i32 2\n}\n"
        )
        only_f = "define i32 @f() {\nentry:\n  ret i32 1\n}\n"
        assert verify_equivalence(orig, only_f) is False

    def test_swapped_branch_condition(self):
        """Invert a comparison in max() — should produce wrong answers."""
        orig = pipeline_ir("03_if_else.c")
        # Replace 'icmp sgt' with 'icmp slt' to flip the condition
        broken = orig.replace("icmp sgt", "icmp slt")
        # Make sure we actually changed something
        if orig == broken:
            pytest.skip("Replacement did not change the IR — skip")
        assert verify_equivalence(orig, broken) is False


# ---------------------------------------------------------------------------
# Interpreter internals
# ---------------------------------------------------------------------------

class TestInterpreter:
    def test_return_constant(self):
        ir = "define i32 @main() {\nentry:\n  ret i32 42\n}\n"
        module = _parse_ir(ir)
        interp = _Interpreter(module)
        assert interp.call("main", []) == 42

    def test_add_two_params(self):
        ir = (
            "define i32 @add(i32 %a, i32 %b) {\n"
            "entry:\n"
            "  %a.addr = alloca i32\n"
            "  %b.addr = alloca i32\n"
            "  store i32 %a, i32* %a.addr\n"
            "  store i32 %b, i32* %b.addr\n"
            "  %0 = load i32, i32* %a.addr\n"
            "  %1 = load i32, i32* %b.addr\n"
            "  %add = add i32 %0, %1\n"
            "  ret i32 %add\n"
            "}\n"
        )
        module = _parse_ir(ir)
        interp = _Interpreter(module)
        assert interp.call("add", [3, 4]) == 7
        assert interp.call("add", [0, 0]) == 0
        assert interp.call("add", [-1, 1]) == 0

    def test_if_else_max(self):
        """Interpret the max() function from sample 03."""
        ir = read_sample("03_if_else.ll")
        module = _parse_ir(ir)
        interp = _Interpreter(module)
        assert interp.call("max", [10, 5]) == 10
        assert interp.call("max", [3, 7])  == 7
        assert interp.call("max", [4, 4])  == 4

    def test_while_count_to(self):
        """Interpret the count_to() function from sample 04."""
        ir = read_sample("04_while.ll")
        module = _parse_ir(ir)
        interp = _Interpreter(module)
        assert interp.call("count_to", [0]) == 0
        assert interp.call("count_to", [5]) == 5

    def test_function_call(self):
        """Interpret multi-function IR from sample 05."""
        ir = read_sample("05_func_call.ll")
        module = _parse_ir(ir)
        interp = _Interpreter(module)
        assert interp.call("main", []) == 10  # double_it(5) = 10
