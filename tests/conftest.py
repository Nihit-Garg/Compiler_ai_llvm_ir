"""
Shared test setup.

advisor/heuristics.py loads a developer's .env when it is imported, so a real
GEMINI_API_KEY can be in os.environ while the tests run.  No test may call the
live Gemini API: it is slow (about a minute per call), costs quota and makes
results depend on the network.  Tests that need an answer patch
advisor.heuristics._call_gemini themselves, which overrides this guard.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_live_gemini_calls(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(
            "a test reached the live Gemini API; patch advisor.heuristics._call_gemini "
            "or remove GEMINI_API_KEY in the test")
    monkeypatch.setattr("advisor.heuristics._call_gemini", refuse)
