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
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        with patch("advisor.heuristics._call_gemini", return_value='["peephole"]'), \
             patch("verifier.pipeline.verify_equivalence", return_value=False):
            ir = pipeline_ir("02_add.c")
            _, report = apply_optimizations(ir)

        assert report["decision"] == "rejected"
        assert report["rejection_reason"] is not None
        assert report["fallback_passes"] == _default_passes()
        assert report["final_passes"] == _default_passes()

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
