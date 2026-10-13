"""
Tests for the command-line driver (main.py).

Tests cover:
- Default output location: build/ in the working directory, never the source folder
- Exit status when --compile fails
- The final IR check (verify_ir + LLVM) recorded in the report and the exit status
- Advisor source and instruction counts in the written report
- --no-advisor and its alias --no-opt
"""

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import main
from verifier.pipeline import apply_optimizations as real_apply_optimizations


SAMPLES_DIR = Path(__file__).parent / "samples"
SAMPLE = SAMPLES_DIR / "05_func_call.c"


def run_main(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    return main.main()


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def in_tmp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def compile_to(tmp_path: Path, **kwargs):
    """Run compile_source on the 05 sample into tmp_path; return (code, report)."""
    ir_out = tmp_path / "out.ll"
    rpt_out = tmp_path / "out_report.json"
    options = dict(do_compile=False, no_opt=False, quiet=True)
    options.update(kwargs)
    code = main.compile_source(str(SAMPLE), str(ir_out), str(rpt_out), **options)
    report = json.loads(rpt_out.read_text()) if rpt_out.exists() else None
    return code, report


def fake_pipeline(ir_text: str):
    """apply_optimizations stand-in that returns the given IR as the final IR."""
    def fake(raw_ir, use_advisor=True):
        _, report = real_apply_optimizations(raw_ir, use_advisor=use_advisor)
        return ir_text, report
    return fake


# ---------------------------------------------------------------------------
# 1. Default output location
# ---------------------------------------------------------------------------

class TestDefaultOutputLocation:
    def test_outputs_go_to_build_folder(self, monkeypatch, tmp_path, in_tmp, no_key):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        shutil.copy(SAMPLE, src_dir / SAMPLE.name)

        assert run_main(monkeypatch, str(src_dir / SAMPLE.name), "-q") == 0

        assert (tmp_path / "build" / "05_func_call.ll").is_file()
        assert (tmp_path / "build" / "05_func_call_report.json").is_file()
        assert sorted(p.name for p in src_dir.iterdir()) == [SAMPLE.name]

    def test_expected_output_file_is_not_overwritten(self, monkeypatch, in_tmp, no_key):
        expected = SAMPLES_DIR / "05_func_call.ll"
        before = expected.read_bytes()
        stray_report = SAMPLES_DIR / "05_func_call_report.json"

        assert run_main(monkeypatch, str(SAMPLE), "-q") == 0

        assert expected.read_bytes() == before
        assert not stray_report.exists()

    def test_output_and_report_flags_still_work(self, monkeypatch, tmp_path, in_tmp, no_key):
        ir_out = tmp_path / "custom" / "x.ll"
        rpt_out = tmp_path / "other" / "x.json"
        assert run_main(monkeypatch, str(SAMPLE), "-q",
                        "-o", str(ir_out), "-r", str(rpt_out)) == 0
        assert ir_out.is_file() and rpt_out.is_file()
        assert not (tmp_path / "build").exists()


# ---------------------------------------------------------------------------
# 2. Exit status when --compile fails
# ---------------------------------------------------------------------------

class TestCompileExitCode:
    def test_backend_failure_returns_non_zero(self, tmp_path, no_key):
        with patch("backend.codegen.generate_object_code",
                   side_effect=RuntimeError("backend exploded")):
            code, report = compile_to(tmp_path, do_compile=True)
        assert code != 0
        assert "backend exploded" in report["compile_error"]
        assert report["object_output"] is None
        assert not (tmp_path / "out.o").exists()

    def test_successful_compile_returns_zero(self, tmp_path, no_key):
        with patch("backend.codegen.generate_object_code", return_value=b"\x7fOBJ"):
            code, report = compile_to(tmp_path, do_compile=True)
        assert code == 0
        assert report["compile_error"] is None
        assert (tmp_path / "out.o").read_bytes() == b"\x7fOBJ"


# ---------------------------------------------------------------------------
# 3. Final IR check
# ---------------------------------------------------------------------------

class TestFinalIRCheck:
    def test_valid_ir_passes_and_is_recorded(self, tmp_path, no_key):
        code, report = compile_to(tmp_path)
        check = report["final_ir_check"]
        assert code == 0
        assert check["verify_ir"] is True
        assert check["passed"] is True
        assert check["llvm"] in ("passed", "unavailable")

    def test_llvm_is_used_when_installed(self, tmp_path, no_key):
        pytest.importorskip("llvmlite.binding")
        _, report = compile_to(tmp_path)
        assert report["final_ir_check"]["llvm"] == "passed"

    def test_ir_rejected_by_llvm_fails_the_run(self, tmp_path, no_key):
        pytest.importorskip("llvmlite.binding")
        # Has the shape verify_ir() looks for, but %x is never defined.
        bad_ir = "define i32 @main() {\nentry:\n  ret i32 %x\n}\n"
        with patch("main.apply_optimizations", side_effect=fake_pipeline(bad_ir)):
            code, report = compile_to(tmp_path)
        check = report["final_ir_check"]
        assert code != 0
        assert check["verify_ir"] is True
        assert check["llvm"] == "failed"
        assert "%x" in check["llvm_error"]
        assert check["passed"] is False
        assert report["exit_code"] == code

    def test_ir_rejected_by_verify_ir_fails_the_run(self, tmp_path, no_key):
        with patch("main.apply_optimizations", side_effect=fake_pipeline("; empty\n")), \
             patch("main.llvm_check", return_value=("unavailable", None)):
            code, report = compile_to(tmp_path)
        assert code != 0
        assert report["final_ir_check"]["verify_ir"] is False
        assert report["final_ir_check"]["passed"] is False

    def test_failed_check_skips_object_code(self, tmp_path, no_key):
        with patch("main.apply_optimizations", side_effect=fake_pipeline("; empty\n")), \
             patch("backend.codegen.generate_object_code",
                   side_effect=AssertionError("must not compile invalid IR")):
            code, report = compile_to(tmp_path, do_compile=True)
        assert code != 0
        assert report["compile_error"].startswith("skipped")


# ---------------------------------------------------------------------------
# 4 + 5. Advisor source and instruction counts in the written report
# ---------------------------------------------------------------------------

class TestReportFields:
    def test_no_api_key_is_reported(self, tmp_path, no_key):
        _, report = compile_to(tmp_path)
        assert report["advisor_source"] == "no_api_key"
        assert report["advisor_model"] is None

    def test_llm_answer_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        monkeypatch.setenv("ADVISOR_MODEL", "some-model")
        with patch("advisor.heuristics._call_gemini", return_value='["peephole"]'):
            _, report = compile_to(tmp_path)
        assert report["advisor_source"] == "llm"
        assert report["advisor_model"] == "some-model"
        assert report["suggested_passes"] == ["peephole"]

    def test_instruction_counts_are_reported(self, tmp_path, no_key):
        _, report = compile_to(tmp_path)
        assert isinstance(report["instructions_before"], int)
        assert isinstance(report["instructions_after"], int)
        assert 0 < report["instructions_after"] <= report["instructions_before"]


# ---------------------------------------------------------------------------
# 7. --no-advisor and the --no-opt alias
# ---------------------------------------------------------------------------

class TestNoAdvisorFlag:
    @pytest.mark.parametrize("flag", ["--no-advisor", "--no-opt"])
    def test_flag_skips_the_llm(self, monkeypatch, tmp_path, in_tmp, flag):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        with patch("advisor.heuristics._call_gemini",
                   side_effect=AssertionError("LLM must not be called")):
            assert run_main(monkeypatch, str(SAMPLE), "-q", flag) == 0
        report = json.loads((tmp_path / "build" / "05_func_call_report.json").read_text())
        assert report["advisor_source"] == "skipped"
        assert report["final_passes"] == report["suggested_passes"] != []

    def test_both_spellings_set_the_same_option(self):
        parser = main._build_parser()
        assert parser.parse_args(["x.c", "--no-advisor"]).no_advisor is True
        assert parser.parse_args(["x.c", "--no-opt"]).no_advisor is True
        assert parser.parse_args(["x.c"]).no_advisor is False

    def test_help_text_says_optimization_still_runs(self):
        text = main._build_parser().format_help()
        assert "--no-advisor" in text
        assert "does not disable optimization" in " ".join(text.split())



# ---------------------------------------------------------------------------
# Output that cannot encode the banner (Windows pipes and redirects)
# ---------------------------------------------------------------------------

def run_cli_cp1252(cwd: Path, *args: str):
    """Run main.py in a subprocess whose stdout/stderr are cp1252 pipes."""
    import os
    import subprocess
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    env.pop("PYTHONUTF8", None)
    env.pop("GEMINI_API_KEY", None)
    script = Path(main.__file__).resolve()
    return subprocess.run(
        [sys.executable, "-X", "utf8=0", str(script), *args],
        cwd=cwd, env=env, capture_output=True, timeout=120,
    )


class TestNarrowOutputEncoding:
    """cp1252 has no box-drawing characters; printing them used to crash main.py."""

    def test_piped_output_does_not_crash(self, tmp_path):
        result = run_cli_cp1252(tmp_path, str(SAMPLE))
        stderr = result.stderr.decode("cp1252", "replace")
        assert result.returncode == 0, stderr
        assert "Traceback" not in stderr
        assert "MiniC Compiler" in result.stdout.decode("cp1252")

    def test_error_message_on_stderr_does_not_crash(self, tmp_path):
        result = run_cli_cp1252(tmp_path, str(tmp_path / "missing.c"))
        stderr = result.stderr.decode("cp1252", "replace")
        assert result.returncode == 1
        assert "Traceback" not in stderr
        assert "File not found" in stderr


# ---------------------------------------------------------------------------
# File encoding: IR, report and source are read and written as UTF-8
# ---------------------------------------------------------------------------

KEEP_SOURCE = "int f(int a) { return a * 1 + 0; }\n"


def cp1252_default_open(calls):
    """
    An open() that behaves like Windows with a cp1252 locale: text files
    opened without an explicit encoding get cp1252.  Every call is recorded.
    """
    import builtins

    def fake_open(file, mode="r", *args, **kwargs):
        calls.append((str(file), mode, kwargs.get("encoding")))
        if "b" not in mode and "encoding" not in kwargs and len(args) < 2:
            kwargs["encoding"] = "cp1252"
        return builtins.open(file, mode, *args, **kwargs)
    return fake_open


class TestFileEncoding:
    def test_writes_ir_for_program_the_peephole_rewrites(self, monkeypatch, tmp_path, in_tmp, no_key):
        (tmp_path / "keep.c").write_text(KEEP_SOURCE, encoding="utf-8")
        calls = []
        monkeypatch.setattr(main, "open", cp1252_default_open(calls), raising=False)

        assert run_main(monkeypatch, "keep.c", "-q") == 0

        ir = (tmp_path / "build" / "keep.ll").read_text(encoding="utf-8")
        assert "define i32 @f(i32 %a)" in ir
        assert "; peephole" in ir
        report = json.loads((tmp_path / "build" / "keep_report.json").read_text(encoding="utf-8"))
        assert report["exit_code"] == 0

    def test_every_text_file_is_opened_as_utf8(self, monkeypatch, tmp_path, in_tmp, no_key):
        (tmp_path / "keep.c").write_text(KEEP_SOURCE, encoding="utf-8")
        calls = []
        monkeypatch.setattr(main, "open", cp1252_default_open(calls), raising=False)

        assert run_main(monkeypatch, "keep.c", "-q") == 0

        text_opens = [c for c in calls if "b" not in c[1]]
        assert {Path(c[0]).name for c in text_opens} == {"keep.c", "keep.ll", "keep_report.json"}
        assert all(c[2] == "utf-8" for c in text_opens), text_opens
