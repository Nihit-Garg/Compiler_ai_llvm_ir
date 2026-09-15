"""
Unit tests for the LLM Advisor (advisor/heuristics.py).

Tests cover:
- Response parsing with valid, partial, and invalid LLM outputs
- Fallback behaviour when GEMINI_API_KEY is not set
- Prompt contents (must include IR text and all valid pass names)
- End-to-end with a mocked Gemini response
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from advisor.heuristics import (
    get_optimization_advice,
    _build_prompt,
    _parse_response,
    _default_passes,
    _VALID_PASSES,
)


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_valid_json_array(self):
        resp = '["constant_folding", "peephole"]'
        assert _parse_response(resp) == ["constant_folding", "peephole"]

    def test_all_valid_passes(self):
        resp = str(_VALID_PASSES).replace("'", '"')
        result = _parse_response(resp)
        assert set(result) == set(_VALID_PASSES)

    def test_filters_unknown_names(self):
        resp = '["constant_folding", "fancy_new_pass", "peephole"]'
        result = _parse_response(resp)
        assert result == ["constant_folding", "peephole"]

    def test_deduplication(self):
        resp = '["peephole", "peephole", "constant_folding"]'
        result = _parse_response(resp)
        assert result == ["peephole", "constant_folding"]

    def test_empty_array(self):
        assert _parse_response("[]") == []

    def test_no_json_array(self):
        assert _parse_response("I suggest using dead code elimination.") == []

    def test_malformed_json(self):
        assert _parse_response('["constant_folding", ') == []

    def test_non_string_elements_ignored(self):
        resp = '[1, "peephole", null]'
        assert _parse_response(resp) == ["peephole"]

    def test_array_embedded_in_prose(self):
        resp = 'Sure! Here is my suggestion: ["dce", "peephole"]. Let me know.'
        # "dce" is not a valid pass name
        assert _parse_response(resp) == ["peephole"]

    def test_valid_embedded_in_prose(self):
        resp = 'Recommendation: ["constant_folding", "dead_code_elimination"]'
        assert _parse_response(resp) == ["constant_folding", "dead_code_elimination"]


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_prompt_contains_ir(self):
        ir = "define i32 @main() {\nentry:\n  ret i32 0\n}\n"
        prompt = _build_prompt(ir)
        assert ir.strip() in prompt

    def test_prompt_contains_all_valid_passes(self):
        ir = "define i32 @f() {\nentry:\n  ret i32 1\n}\n"
        prompt = _build_prompt(ir)
        for pass_name in _VALID_PASSES:
            assert pass_name in prompt

    def test_prompt_instructs_json_only(self):
        ir = "define i32 @f() {\nentry:\n  ret i32 1\n}\n"
        prompt = _build_prompt(ir)
        assert "JSON" in prompt or "json" in prompt.lower()


# ---------------------------------------------------------------------------
# _default_passes
# ---------------------------------------------------------------------------

class TestDefaultPasses:
    def test_returns_all_valid_passes(self):
        defaults = _default_passes()
        assert set(defaults) == set(_VALID_PASSES)
        assert len(defaults) == len(_VALID_PASSES)

    def test_returns_list_not_same_object(self):
        """Ensure mutations don't affect the original."""
        d1 = _default_passes()
        d2 = _default_passes()
        d1.append("extra")
        assert "extra" not in d2


# ---------------------------------------------------------------------------
# get_optimization_advice — fallback on no API key
# ---------------------------------------------------------------------------

class TestGetOptimizationAdvice:
    def test_no_api_key_returns_default(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ir = "define i32 @main() {\nentry:\n  ret i32 42\n}\n"
        result = get_optimization_advice(ir)
        assert result == _default_passes()

    def test_empty_api_key_returns_default(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "   ")
        ir = "define i32 @main() {\nentry:\n  ret i32 42\n}\n"
        result = get_optimization_advice(ir)
        assert result == _default_passes()

    def test_with_mocked_gemini_success(self, monkeypatch):
        """Simulate a successful Gemini API call returning a valid response."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

        mock_response = MagicMock()
        mock_response.text = '["peephole", "dead_code_elimination"]'

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        with patch("advisor.heuristics._call_gemini", return_value=mock_response.text):
            result = get_optimization_advice(
                "define i32 @main() {\nentry:\n  ret i32 42\n}\n"
            )

        assert result == ["peephole", "dead_code_elimination"]

    def test_with_mocked_gemini_api_failure(self, monkeypatch):
        """If _call_gemini returns None (API failure), fall back to defaults."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

        with patch("advisor.heuristics._call_gemini", return_value=None):
            result = get_optimization_advice(
                "define i32 @main() {\nentry:\n  ret i32 42\n}\n"
            )

        assert result == _default_passes()

    def test_with_mocked_gemini_unparseable_response(self, monkeypatch):
        """If Gemini returns text with no valid passes, fall back to defaults."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

        with patch("advisor.heuristics._call_gemini", return_value="Sorry, I cannot help."):
            result = get_optimization_advice(
                "define i32 @main() {\nentry:\n  ret i32 42\n}\n"
            )

        assert result == _default_passes()

    def test_result_is_always_a_list(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = get_optimization_advice("define i32 @f() {\nentry:\n  ret i32 0\n}\n")
        assert isinstance(result, list)

    def test_result_only_contains_valid_passes(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
        with patch("advisor.heuristics._call_gemini",
                   return_value='["peephole", "invalid_pass", "constant_folding"]'):
            result = get_optimization_advice("define i32 @f() {\nentry:\n  ret i32 0\n}\n")
        for p in result:
            assert p in _VALID_PASSES
