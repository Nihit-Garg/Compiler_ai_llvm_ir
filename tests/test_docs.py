"""
Consistency checks between the code and the documentation / toolchain files.

These catch the README, requirements.txt, CI and SPEC drifting away from
what the code actually does.
"""

import re
from pathlib import Path

import pytest

import main
from advisor.heuristics import _DEFAULT_MODEL

ROOT = Path(__file__).resolve().parent.parent


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# One default model name everywhere
# ---------------------------------------------------------------------------

def test_default_model_is_the_chosen_one():
    assert _DEFAULT_MODEL == "gemini-3.5-flash-lite"


@pytest.mark.parametrize("rel, pattern", [
    ("advisor/heuristics.py", r'Defaults to "([\w.\-]+)"'),
    ("main.py",               r"ADVISOR_MODEL\s+Gemini model name \(default: ([\w.\-]+)\)"),
    (".env.example",          r"override the Gemini model \(default: ([\w.\-]+)\)"),
    ("docs/SPEC.md",          r"\| `ADVISOR_MODEL` \| No \| `([\w.\-]+)` \|"),
])
def test_documented_default_model_matches_code(rel, pattern):
    match = re.search(pattern, read(rel))
    assert match, f"no default model mentioned in {rel}"
    assert match.group(1) == _DEFAULT_MODEL, rel


def test_advisor_uses_default_model_when_unset(monkeypatch):
    from unittest.mock import patch
    from advisor.heuristics import get_advice_with_source
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.delenv("ADVISOR_MODEL", raising=False)
    with patch("advisor.heuristics._call_gemini", return_value='["peephole"]') as call:
        _, info = get_advice_with_source("define i32 @f() {\nentry:\n  ret i32 0\n}\n")
    assert call.call_args[0][2] == _DEFAULT_MODEL
    assert info["model"] == _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Toolchain: no system LLVM, llvmlite range, install command
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel", [
    "README.md", "requirements.txt", ".github/workflows/ci.yml",
    "backend/codegen.py", "main.py",
])
def test_no_system_llvm_14_requirement(rel):
    text = read(rel)
    assert "llvm-14" not in text
    assert "LLVM 14" not in text


def test_llvmlite_is_a_minimum_not_a_pin():
    lines = [l.strip() for l in read("requirements.txt").splitlines()
             if l.strip().startswith("llvmlite")]
    assert lines == ["llvmlite>=0.40.0"]


def test_readme_installs_from_requirements():
    readme = read("README.md")
    assert "pip install -r requirements.txt" in readme
    assert "pip install llvmlite pytest" not in readme


def test_readme_python_range_matches_ci():
    readme = read("README.md")
    ci = read(".github/workflows/ci.yml")
    assert "3.10 or higher" not in readme
    assert "**Python**: 3.10, 3.11, 3.12 or 3.13" in readme
    assert 'python-version: "3.10"' in ci and 'llvmlite: "llvmlite==0.40.0"' in ci
    assert 'python-version: "3.12"' in ci


def test_readme_documents_build_folder():
    readme = read("README.md")
    assert "build/" in readme
    assert "overwrites the first unless you give `-o`" in readme
    assert main.DEFAULT_BUILD_DIR == "build"


# ---------------------------------------------------------------------------
# .env.example is a template: it must never carry a real key
# ---------------------------------------------------------------------------

def test_env_example_has_only_a_placeholder_key():
    keys = [l.strip() for l in read(".env.example").splitlines()
            if l.strip().startswith("GEMINI_API_KEY")]
    assert keys == ["GEMINI_API_KEY=your-gemini-api-key-here"]
