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


# ---------------------------------------------------------------------------
# Regression tests: the verifier must not approve what it could not check
# ---------------------------------------------------------------------------

import verifier.check as check
from verifier.check import equivalence_failure_reason, _generate_inputs


def compile_src(src: str) -> str:
    from frontend.lexer import lex
    from frontend.parser import parse
    from frontend.semantic import analyze
    from ir.generator import generate_ir
    return generate_ir(analyze(parse(lex(src))))


def ret_const(value) -> str:
    return f"define i32 @main() {{\nentry:\n  ret i32 {value}\n}}\n"


class TestThirtyTwoBitArithmetic:
    def test_sdiv_truncates_toward_zero(self):
        orig = compile_src("int main() { return (0 - 7) / 2; }")
        assert verify_equivalence(orig, ret_const(-3)) is True
        assert verify_equivalence(orig, ret_const(-4)) is False

    def test_multiplication_wraps(self):
        orig = compile_src("int main() { return (65536 * 65536) / 2; }")
        assert verify_equivalence(orig, ret_const(0)) is True
        assert verify_equivalence(orig, ret_const(2147483648)) is False

    def test_out_of_range_literal_is_wrapped(self):
        assert verify_equivalence(ret_const(4294967296), ret_const(0)) is True


class TestInconclusiveRuns:
    LOOP_THEN = (
        "int main() {{ int i = 0; while (i < 1000) {{ i = i + 1; }} return {}; }}"
    )

    def test_step_limit_on_both_sides_is_not_equivalence(self, monkeypatch):
        """Two different programs that both outrun the step limit must not pass."""
        monkeypatch.setattr(check, "_MAX_STEPS", 500)
        a = compile_src(self.LOOP_THEN.format(1))
        b = compile_src(self.LOOP_THEN.format(2))
        assert verify_equivalence(a, b) is False
        assert "inconclusive" in equivalence_failure_reason(a, b)

    def test_same_programs_pass_once_the_limit_is_high_enough(self):
        a = compile_src(self.LOOP_THEN.format(1))
        assert verify_equivalence(a, a) is True

    def test_different_results_after_a_long_loop_are_caught(self):
        a = compile_src(self.LOOP_THEN.format(1))
        b = compile_src(self.LOOP_THEN.format(2))
        assert verify_equivalence(a, b) is False

    def test_runaway_recursion_does_not_crash(self):
        """fact(-1) never reaches its n == 0 base case; that input is skipped."""
        ir = compile_src(
            "int fact(int n) { if (n == 0) { return 1; } return n * fact(n - 1); }"
        )
        assert verify_equivalence(ir, ir) is True

    def test_function_that_never_finishes_is_not_verified(self):
        ir = compile_src(
            "int spin() { int i = 0; while (i == 0) { i = 0; } return i; }"
        )
        assert verify_equivalence(ir, ir) is False


class TestTrapsAndMalformedIr:
    DIV = "define i32 @f(i32 %a) {\nentry:\n  %q = sdiv i32 1, %a\n  ret i32 %q\n}\n"

    def test_trap_matches_trap(self):
        assert verify_equivalence(self.DIV, self.DIV) is True

    def test_trap_does_not_match_value(self):
        no_trap = "define i32 @f(i32 %a) {\nentry:\n  ret i32 0\n}\n"
        assert verify_equivalence(self.DIV, no_trap) is False

    def test_unknown_instruction_is_not_ignored(self):
        good = "define i32 @main() {\nentry:\n  ret i32 1\n}\n"
        odd = "define i32 @main() {\nentry:\n  %x = xor i32 1, 1\n  ret i32 1\n}\n"
        assert verify_equivalence(good, odd) is False

    def test_undefined_register_is_rejected(self):
        good = "define i32 @main() {\nentry:\n  ret i32 1\n}\n"
        bad = "define i32 @main() {\nentry:\n  ret i32 %nope\n}\n"
        assert verify_equivalence(good, bad) is False

    def test_reason_is_none_for_equivalent_ir(self):
        good = "define i32 @main() {\nentry:\n  ret i32 1\n}\n"
        assert equivalence_failure_reason(good, good) is None


class TestInputCoverage:
    @pytest.mark.parametrize("n_params", [1, 2, 3, 4, 5])
    def test_every_parameter_sees_every_test_value(self, n_params):
        inputs = _generate_inputs(n_params)
        assert 0 < len(inputs) <= check._MAX_TESTS
        for position in range(n_params):
            seen = {args[position] for args in inputs}
            assert seen == set(check._TEST_VALUES), (
                f"parameter {position} of {n_params} only saw {sorted(seen)}")


# ---------------------------------------------------------------------------
# count_instructions and llvm_check
# ---------------------------------------------------------------------------

from verifier.check import count_instructions, llvm_check


class TestCountInstructions:
    def test_counts_instructions_not_labels_or_comments(self):
        ir = ("; comment\n"
              "define i32 @f(i32 %a) {\n"
              "entry:\n"
              "  %x = add i32 %a, 1   ; inline comment\n"
              "  br label %next\n"
              "\n"
              "next:\n"
              "  ret i32 %x\n"
              "}\n")
        assert count_instructions(ir) == 3

    def test_counts_across_functions(self):
        ir = ("define i32 @f() {\nentry:\n  ret i32 1\n}\n\n"
              "define i32 @g() {\nentry:\n  %y = add i32 1, 2\n  ret i32 %y\n}\n")
        assert count_instructions(ir) == 3

    def test_empty_ir_has_no_instructions(self):
        assert count_instructions("") == 0


class TestLLVMCheck:
    def test_unavailable_without_llvmlite(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def no_llvmlite(name, *args, **kwargs):
            if name.startswith("llvmlite"):
                raise ImportError("no llvmlite here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_llvmlite)
        assert llvm_check("define i32 @f() {\nentry:\n  ret i32 0\n}\n") == ("unavailable", None)

    def test_valid_ir_passes(self):
        pytest.importorskip("llvmlite.binding")
        assert llvm_check("define i32 @f() {\nentry:\n  ret i32 0\n}\n") == ("passed", None)

    def test_invalid_ir_fails_with_llvm_message(self):
        pytest.importorskip("llvmlite.binding")
        status, error = llvm_check("define i32 @f() {\nentry:\n  ret i32 %x\n}\n")
        assert status == "failed"
        assert "%x" in error and "\n" not in error
