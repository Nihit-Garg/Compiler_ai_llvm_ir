"""
Integration tests for the fallback orchestrator (verifier/pipeline.py).

Tests cover:
- apply_optimizations returns the correct tuple shape
- The accepted path: when the LLM suggests valid passes and the verifier agrees
- The rejected path: when the verifier rejects → fallback is applied
- The report dict contains all expected keys and correct values
- End-to-end without a live LLM (GEMINI_API_KEY not set → default passes used)
"""

import os
import pytest
from unittest.mock import patch

from verifier.pipeline import apply_optimizations
from advisor.heuristics import _default_passes, _VALID_PASSES


SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


def pipeline_ir(c_filename: str) -> str:
    from frontend.lexer import lex
    from frontend.parser import parse
    from frontend.semantic import analyze
    from ir.generator import generate_ir
    with open(os.path.join(SAMPLES_DIR, c_filename)) as f:
        src = f.read()
    return generate_ir(analyze(parse(lex(src))))


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------

class TestReturnShape:
    def test_returns_tuple_of_str_and_dict(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = pipeline_ir("01_return.c")
        result = apply_optimizations(ir)
        assert isinstance(result, tuple) and len(result) == 2
        final_ir, report = result
        assert isinstance(final_ir, str)
        assert isinstance(report, dict)

    def test_report_has_required_keys(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = pipeline_ir("01_return.c")
        _, report = apply_optimizations(ir)
        required = {"suggested_passes", "decision", "rejection_reason",
                    "fallback_passes", "final_passes"}
        assert required.issubset(report.keys())

    def test_final_ir_is_non_empty(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = pipeline_ir("02_add.c")
        final_ir, _ = apply_optimizations(ir)
        assert len(final_ir.strip()) > 0
        assert "define i32" in final_ir

    def test_final_ir_contains_ret(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = pipeline_ir("02_add.c")
        final_ir, _ = apply_optimizations(ir)
        assert "ret i32" in final_ir


# ---------------------------------------------------------------------------
# No API key → default passes accepted
# ---------------------------------------------------------------------------

class TestNoApiKey:
    def test_decision_is_accepted(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = pipeline_ir("01_return.c")
        _, report = apply_optimizations(ir)
        # Default passes always produce equivalent IR → accepted
        assert report["decision"] == "accepted"

    def test_suggested_passes_are_defaults(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = pipeline_ir("02_add.c")
        _, report = apply_optimizations(ir)
        assert report["suggested_passes"] == _default_passes()

    def test_rejection_reason_is_none(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = pipeline_ir("03_if_else.c")
        _, report = apply_optimizations(ir)
        assert report["rejection_reason"] is None

    def test_final_passes_match_suggested(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = pipeline_ir("04_while.c")
        _, report = apply_optimizations(ir)
        assert report["final_passes"] == report["suggested_passes"]


# ---------------------------------------------------------------------------
# Mocked LLM success → valid suggestion accepted
# ---------------------------------------------------------------------------

class TestMockedAccepted:
    def test_valid_suggestion_accepted(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        suggestion = ["peephole", "dead_code_elimination"]
        with patch("advisor.heuristics._call_gemini",
                   return_value=str(suggestion).replace("'", '"')):
            ir = pipeline_ir("02_add.c")
            _, report = apply_optimizations(ir)
        assert report["decision"] == "accepted"
        assert report["suggested_passes"] == suggestion
        assert report["final_passes"] == suggestion
        assert report["fallback_passes"] is None

    def test_all_samples_with_peephole_only(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        with patch("advisor.heuristics._call_gemini", return_value='["peephole"]'):
            for n in ['01_return', '02_add', '03_if_else', '04_while', '05_func_call']:
                ir = pipeline_ir(f"{n}.c")
                _, report = apply_optimizations(ir)
                assert report["decision"] == "accepted", \
                    f"peephole-only was rejected for {n}.c"


# ---------------------------------------------------------------------------
# Mocked bad verifier → fallback triggered
# ---------------------------------------------------------------------------

class TestMockedRejected:
    def test_rejected_triggers_fallback(self, monkeypatch):
        """Suggestion rejected, default order verified → default order is used."""
        from verifier.check import verify_equivalence as real_verify
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        calls = []

        def reject_first(orig, opt):
            calls.append(opt)
            if len(calls) == 1:
                return False              # reject the suggestion only
            return real_verify(orig, opt)

        with patch("advisor.heuristics._call_gemini", return_value='["peephole"]'), \
             patch("verifier.pipeline.verify_equivalence", side_effect=reject_first):
            ir = pipeline_ir("02_add.c")
            _, report = apply_optimizations(ir)

        assert report["decision"] == "rejected"
        assert report["rejection_reason"] is not None
        assert report["fallback_passes"] == _default_passes()
        assert report["fallback_decision"] == "accepted"
        assert report["final_passes"] == _default_passes()

    def test_unverified_fallback_returns_original_ir(self, monkeypatch):
        """If the default order cannot be verified either, nothing is applied."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        with patch("advisor.heuristics._call_gemini", return_value='["peephole"]'), \
             patch("verifier.pipeline.verify_equivalence", return_value=False):
            ir = pipeline_ir("05_func_call.c")
            final_ir, report = apply_optimizations(ir)

        assert final_ir == ir
        assert report["decision"] == "rejected"
        assert report["fallback_passes"] == _default_passes()
        assert report["fallback_decision"] == "rejected"
        assert report["fallback_rejection_reason"] is not None
        assert report["final_passes"] == []

    def test_verifier_exception_is_a_rejection_not_a_crash(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with patch("verifier.pipeline.verify_equivalence",
                   side_effect=RecursionError("boom")):
            ir = pipeline_ir("05_func_call.c")
            final_ir, report = apply_optimizations(ir)

        assert final_ir == ir
        assert "Verifier error" in report["rejection_reason"]
        assert report["final_passes"] == []

    def test_rejected_final_ir_still_has_ret(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        with patch("advisor.heuristics._call_gemini", return_value='["peephole"]'), \
             patch("verifier.pipeline.verify_equivalence", return_value=False):
            ir = pipeline_ir("02_add.c")
            final_ir, _ = apply_optimizations(ir)
        assert "ret i32" in final_ir

    def test_rejected_final_ir_is_equivalent_to_original(self, monkeypatch):
        """Even after a rejection, the fallback IR must be equivalent to original."""
        from verifier.check import verify_equivalence
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        # Patch _call_gemini to suggest empty passes, then patch verify_equivalence
        # to reject the first call (candidate) but allow us to check the final IR ourselves
        calls = []
        real_verify = verify_equivalence

        def mock_verify(orig, opt):
            calls.append((orig, opt))
            if len(calls) == 1:
                return False  # Reject the candidate
            return real_verify(orig, opt)

        with patch("advisor.heuristics._call_gemini", return_value='["peephole"]'), \
             patch("verifier.pipeline.verify_equivalence", side_effect=mock_verify):
            ir = pipeline_ir("05_func_call.c")
            final_ir, report = apply_optimizations(ir)

        # The fallback IR should be equivalent to the original
        assert real_verify(ir, final_ir) is True


# ---------------------------------------------------------------------------
# End-to-end: all 5 samples, no LLM key
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_all_samples_produce_valid_ir(self, monkeypatch):
        from verifier.check import verify_ir
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        for n in ['01_return', '02_add', '03_if_else', '04_while', '05_func_call']:
            ir = pipeline_ir(f"{n}.c")
            final_ir, report = apply_optimizations(ir)
            assert verify_ir(final_ir), f"final IR for {n}.c failed verify_ir"
            assert report["decision"] in ("accepted", "rejected")


# ---------------------------------------------------------------------------
# Regression tests: programs outside the five samples
# ---------------------------------------------------------------------------

def compile_src(src: str) -> str:
    from frontend.lexer import lex
    from frontend.parser import parse
    from frontend.semantic import analyze
    from ir.generator import generate_ir
    return generate_ir(analyze(parse(lex(src))))


class TestRegressions:
    def test_negative_division_is_folded_correctly_and_accepted(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = compile_src("int main() { return (0 - 7) / 2; }")
        final_ir, report = apply_optimizations(ir)
        assert report["decision"] == "accepted"
        assert "ret i32 -3" in final_ir

    def test_overflowing_multiplication_is_folded_correctly(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = compile_src("int main() { return (65536 * 65536) / 2; }")
        final_ir, report = apply_optimizations(ir)
        assert report["decision"] == "accepted"
        assert "ret i32 0" in final_ir

    def test_recursive_function_does_not_crash_the_pipeline(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = compile_src(
            "int fact(int n) { if (n == 0) { return 1; } return n * fact(n - 1); }"
        )
        final_ir, report = apply_optimizations(ir)
        assert report["decision"] == "accepted"
        assert "define i32 @fact" in final_ir

    def test_unverifiable_program_is_left_unoptimized(self, monkeypatch):
        """A function that never finishes cannot be checked, so it is not changed."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = compile_src(
            "int spin() { int i = 0; while (i == 0) { i = 0 * 5; } return i; }"
        )
        final_ir, report = apply_optimizations(ir)
        assert final_ir == ir
        assert report["decision"] == "rejected"
        assert report["fallback_decision"] == "rejected"
        assert "inconclusive" in report["fallback_rejection_reason"]
        assert report["final_passes"] == []


class TestWithoutAdvisor:
    """use_advisor=False (the CLI's --no-opt) skips the LLM, not the verifier."""

    def test_advisor_is_not_consulted(self, monkeypatch):
        # The pipeline now asks the advisor through get_advice_with_source();
        # patching that (and the API call under it) keeps this test meaningful.
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        with patch("verifier.pipeline.get_advice_with_source",
                   side_effect=AssertionError("advisor must not be called")), \
             patch("advisor.heuristics._call_gemini",
                   side_effect=AssertionError("LLM must not be called")):
            ir = pipeline_ir("05_func_call.c")
            final_ir, report = apply_optimizations(ir, use_advisor=False)
        assert report["suggested_passes"] == _default_passes()
        assert report["decision"] == "accepted"
        assert report["final_passes"] == _default_passes()

    def test_default_order_is_still_verified(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with patch("verifier.pipeline.verify_equivalence", return_value=False):
            ir = pipeline_ir("05_func_call.c")
            final_ir, report = apply_optimizations(ir, use_advisor=False)
        assert final_ir == ir
        assert report["decision"] == "rejected"
        assert report["final_passes"] == []


# ---------------------------------------------------------------------------
# Explicit empty suggestion: no passes, original IR
# ---------------------------------------------------------------------------

class TestEmptySuggestion:
    def test_empty_answer_applies_no_passes(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        with patch("advisor.heuristics._call_gemini", return_value="[]"):
            ir = compile_src("int main() { return 2 + 3; }")
            final_ir, report = apply_optimizations(ir)
        assert final_ir == ir
        assert report["suggested_passes"] == []
        assert report["decision"] == "accepted"
        assert report["final_passes"] == []
        assert report["fallback_passes"] is None
        assert report["advisor_source"] == "llm"

    def test_unverifiable_program_is_not_sent_to_the_fallback(self, monkeypatch):
        """[] needs no verification, so an inconclusive program is not 'rejected'."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        with patch("advisor.heuristics._call_gemini", return_value="[]"), \
             patch("verifier.pipeline.verify_equivalence",
                   side_effect=AssertionError("nothing to verify")):
            ir = compile_src(
                "int spin() { int i = 0; while (i == 0) { i = 0 * 5; } return i; }"
            )
            final_ir, report = apply_optimizations(ir)
        assert final_ir == ir
        assert report["decision"] == "accepted"
        assert report["fallback_decision"] is None


# ---------------------------------------------------------------------------
# Advisor source and instruction counts in the report
# ---------------------------------------------------------------------------

class TestAdvisorSourceInReport:
    def test_no_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        _, report = apply_optimizations(pipeline_ir("02_add.c"))
        assert report["advisor_source"] == "no_api_key"
        assert report["advisor_model"] is None

    def test_llm(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        monkeypatch.setenv("ADVISOR_MODEL", "test-model")
        with patch("advisor.heuristics._call_gemini", return_value='["peephole"]'):
            _, report = apply_optimizations(pipeline_ir("02_add.c"))
        assert report["advisor_source"] == "llm"
        assert report["advisor_model"] == "test-model"

    def test_api_error(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        with patch("advisor.heuristics._call_gemini", return_value=None), \
             patch("advisor.heuristics._genai_installed", return_value=True):
            _, report = apply_optimizations(pipeline_ir("02_add.c"))
        assert report["advisor_source"] == "api_error"
        assert report["suggested_passes"] == _default_passes()

    def test_unusable_response(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        with patch("advisor.heuristics._call_gemini", return_value="no idea"):
            _, report = apply_optimizations(pipeline_ir("02_add.c"))
        assert report["advisor_source"] == "unusable_response"

    def test_skipped(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        _, report = apply_optimizations(pipeline_ir("02_add.c"), use_advisor=False)
        assert report["advisor_source"] == "skipped"
        assert report["advisor_model"] is None


class TestInstructionCounts:
    def test_counts_show_the_improvement(self, monkeypatch):
        from verifier.check import count_instructions
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = compile_src("int main() { int x = 2 + 3; return x * 1; }")
        final_ir, report = apply_optimizations(ir)
        assert report["instructions_before"] == count_instructions(ir)
        assert report["instructions_after"] == count_instructions(final_ir)
        assert report["instructions_after"] < report["instructions_before"]

    def test_counts_equal_when_ir_is_left_unchanged(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with patch("verifier.pipeline.verify_equivalence", return_value=False):
            _, report = apply_optimizations(pipeline_ir("05_func_call.c"))
        assert report["final_passes"] == []
        assert report["instructions_after"] == report["instructions_before"]
