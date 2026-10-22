"""
Tests for main.py --show-steps and --interactive (show_steps.py).

Tests cover:
- Every stage's output is printed: source, tokens, AST, symbols, raw IR,
  each pass, the verifier table, the decision and the final IR
- A rejected suggestion shows the differing inputs and the fallback
- --show-steps does not change the IR or the report
- --quiet still prints only the IR, even with --show-steps
- --interactive and "-" read the program from stdin and write build/input.*
"""

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import main
import verifier.pipeline as pipeline


B8 = Path(__file__).parent.parent / "evaluation" / "benchmarks" / "b8_simplify_then_fold.c"


def run_main(monkeypatch, *argv: str, stdin: str = None) -> int:
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    return main.main()


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def in_tmp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def fake_llm(passes):
    return patch.object(pipeline, "get_advice_with_source",
                        return_value=(passes, {"source": "llm", "model": "simulated"}))


# ---------------------------------------------------------------------------
# Every stage is shown
# ---------------------------------------------------------------------------

def test_every_stage_is_printed(monkeypatch, capsys, in_tmp):
    with fake_llm(["peephole", "constant_folding", "dead_code_elimination"]):
        assert run_main(monkeypatch, str(B8), "--show-steps") == 0
    out = capsys.readouterr().out
    for heading in ["Input source code", "Step 1a: Lexer output", "Step 1b: Parser output",
                    "Step 2: Semantic analysis", "Step 3: Generated LLVM IR",
                    "Step 4a: LLM advisor", "Step 4b: Applying the suggested passes",
                    "Step 4c: Verifier", "Step 4d: Decision", "Step 4e: Final IR check",
                    "Final optimized IR"]:
        assert heading in out, heading
    assert "return (a * 0 + 2) * 3;" in out                  # source
    assert "STAR" in out and "NUMBER" in out                 # tokens
    assert "BinaryOp '*'" in out and "Identifier a" in out   # AST
    assert "a          parameter" in out                     # symbols
    assert "after pass 1: peephole  (7 → 7 instructions)" in out
    assert "after pass 3: dead_code_elimination  (7 → 3 instructions)" in out
    assert "7 matched, 0 skipped, 0 different" in out
    assert "ret i32 6" in out


def test_rejected_suggestion_shows_difference_and_fallback(monkeypatch, capsys, in_tmp):
    broken = lambda ir: ir.replace(" sub i32 ", " add i32 ")
    with patch.dict(pipeline._PASS_FN, {"broken_pass": broken}), \
         fake_llm(["peephole", "broken_pass"]):
        run_main(monkeypatch, "-", "-s",
                 stdin="int diff(int a, int b) { return a - b + 0; }")
    out = capsys.readouterr().out
    assert "(-7, -7)           0              -14            ✗ DIFFERENT" in out
    assert "more input(s) not shown" in out
    assert "Fallback: applying the default order one by one" in out
    assert "Suggestion : REJECTED" in out
    assert "Fallback   : ACCEPTED" in out


def test_compile_shows_backend(monkeypatch, capsys, in_tmp, no_key):
    pytest.importorskip("llvmlite")
    assert run_main(monkeypatch, str(B8), "-s", "--compile") == 0
    assert "Step 5: Backend (object code)" in capsys.readouterr().out


def test_semantic_error_stops_after_the_ast(monkeypatch, capsys, in_tmp):
    assert run_main(monkeypatch, "-i", "-s", stdin="int f() { return z; }") == 1
    captured = capsys.readouterr()
    assert "Step 1b: Parser output" in captured.out
    assert "Step 2: Semantic analysis" not in captured.out
    assert "Undeclared symbol 'z'" in captured.err


# ---------------------------------------------------------------------------
# Showing the steps changes nothing
# ---------------------------------------------------------------------------

def test_show_steps_does_not_change_ir_or_report(monkeypatch, in_tmp, no_key):
    run_main(monkeypatch, str(B8), "-o", "plain.ll", "-r", "plain.json")
    run_main(monkeypatch, str(B8), "-s", "-o", "shown.ll", "-r", "shown.json")
    assert Path("plain.ll").read_text() == Path("shown.ll").read_text()
    plain = json.loads(Path("plain.json").read_text())
    shown = json.loads(Path("shown.json").read_text())
    for r in (plain, shown):
        r.pop("elapsed_s"); r.pop("ir_output")
    assert plain == shown


def test_quiet_wins_over_show_steps(monkeypatch, capsys, in_tmp, no_key):
    assert run_main(monkeypatch, "-", "-q", "-s", stdin="int f() { return 1 + 2; }") == 0
    assert capsys.readouterr().out == "define i32 @f() {\nentry:\n  ret i32 3\n}\n"


# ---------------------------------------------------------------------------
# Reading the program from the terminal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--interactive", "-i", "-"])
def test_program_from_stdin(monkeypatch, in_tmp, no_key, flag):
    assert run_main(monkeypatch, flag, "-q", stdin="int main() { return 42; }") == 0
    assert Path("build/input.ll").is_file()
    report = json.loads(Path("build/input_report.json").read_text())
    assert report["source_file"] == "<stdin>"


def test_no_source_and_no_interactive_is_an_error(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch)
    assert exc.value.code == 2
    assert "--interactive" in capsys.readouterr().err
