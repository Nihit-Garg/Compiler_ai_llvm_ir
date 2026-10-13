"""
Tests for the evaluation script (evaluation/run_eval.py).

They need no API key: LLM answers are simulated or replayed from a dict.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from advisor.heuristics import _default_passes
from evaluation import run_eval


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Safety net: every case ends with correct IR, for the stated reason
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cases():
    import os
    old = os.environ.pop("GEMINI_API_KEY", None)
    try:
        return {c["case"][0]: c for c in run_eval.safety_cases()}
    finally:
        if old is not None:
            os.environ["GEMINI_API_KEY"] = old


def test_faulty_suggestion_is_rejected_and_fallback_used(cases):
    c = cases["a"]
    assert c["decision"] == "rejected"
    # The verifier names the input on which the two versions disagree.
    assert "original returned" in c["rejection_reason"]
    assert c["fallback_decision"] == "accepted"
    assert "broken_pass" not in c["final_passes"]
    assert c["final_equivalent"]


def test_uncheckable_program_is_left_unchanged(cases):
    c = cases["b"]
    assert c["decision"] == "rejected"
    assert c["fallback_decision"] == "rejected"
    assert c["final_passes"] == []
    assert c["ir_unchanged"]


def test_unusable_answer_uses_default_order(cases):
    c = cases["c"]
    assert c["advisor_source"] == "unusable_response"
    assert c["suggested"] == _default_passes()
    assert c["decision"] == "accepted"
    assert c["final_equivalent"]


def test_empty_answer_applies_nothing(cases):
    c = cases["d"]
    assert c["suggested"] == []
    assert c["final_passes"] == []
    assert c["ir_unchanged"]


def test_broken_pass_really_changes_behaviour():
    """Case (a) only proves something if the faulty pass is actually wrong."""
    ir = run_eval.compile_c("int diff(int a, int b) { return a - b; }")
    assert not run_eval.verify_equivalence(ir, run_eval.broken_pass(ir))


def test_broken_pass_is_not_left_registered():
    run_eval.safety_cases()
    assert "broken_pass" not in run_eval.pipeline._PASS_FN


# ---------------------------------------------------------------------------
# Usefulness: best order search and replay of saved answers
# ---------------------------------------------------------------------------

BENCH = Path(run_eval.__file__).parent / "benchmarks"


def test_benchmarks_exist_and_compile():
    paths = sorted(BENCH.glob("*.c"))
    assert len(paths) >= 5
    for p in paths:
        assert "define i32" in run_eval.compile_c(p.read_text(encoding="utf-8"))


def test_best_order_is_never_worse_than_default():
    for p in sorted(BENCH.glob("*.c")):
        ir = run_eval.compile_c(p.read_text(encoding="utf-8"))
        default = run_eval.count_instructions(
            run_eval._apply_pass_sequence(ir, _default_passes()))
        best_n, best_passes = run_eval.best_order(ir)
        assert best_n <= default, p.name


def test_order_matters_on_benchmarks():
    """The benchmarks exist to show that some order beats the fixed default."""
    beaten = 0
    for p in sorted(BENCH.glob("*.c")):
        ir = run_eval.compile_c(p.read_text(encoding="utf-8"))
        default = run_eval.count_instructions(
            run_eval._apply_pass_sequence(ir, _default_passes()))
        beaten += run_eval.best_order(ir)[0] < default
    assert beaten >= 3


def test_saved_answer_is_replayed_without_calling_the_api():
    path = BENCH / "b2_identities.c"
    saved = {"b2_identities": {"suggested": ["peephole", "dead_code_elimination"],
                               "source": "llm", "model": "saved-model", "seconds": 1.0}}
    with patch("advisor.heuristics._call_gemini",
               side_effect=AssertionError("must not call the API")):
        row = run_eval.evaluate_program(path, saved, live=False)
    assert row["llm_source"] == "llm"
    assert row["llm_suggested"] == ["peephole", "dead_code_elimination"]
    assert row["llm_decision"] == "accepted"
    assert row["llm"] < row["default"]


def test_tagged_baseline_is_not_extended_with_new_answers(monkeypatch, tmp_path):
    """Re-running --tag names_only must not ask the LLM about programs added later."""
    import json
    import sys
    tag = "frozen"
    (tmp_path / f"llm_suggestions_{tag}.json").write_text(json.dumps({
        "01_return": {"suggested": [], "source": "llm", "model": "m", "seconds": 1.0}}))
    monkeypatch.setattr(run_eval, "HERE", tmp_path)
    monkeypatch.setattr(run_eval, "PROGRAMS", [
        Path(run_eval.ROOT) / "tests" / "samples" / "01_return.c",
        BENCH / "b8_simplify_then_fold.c"])
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--tag", tag])
    evaluated = []
    real = run_eval.evaluate_program

    def record(path, *args, **kwargs):
        evaluated.append(path.stem)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(run_eval, "evaluate_program", record)
    assert run_eval.main() == 0
    assert evaluated == ["01_return"]
    rows = json.loads((tmp_path / f"results_{tag}.json").read_text())["programs"]
    assert [r["program"] for r in rows] == ["01_return"]
    saved = json.loads((tmp_path / f"llm_suggestions_{tag}.json").read_text())
    assert list(saved) == ["01_return"]


def test_benchmarks_sort_by_number():
    names = [p.stem.split("_")[0] for p in run_eval.PROGRAMS if p.parent == BENCH]
    assert names == sorted(names, key=lambda n: int(n[1:]))
