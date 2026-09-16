"""
Tests for backend/codegen.py.

llvmlite requires LLVM 14 system libraries. Tests are automatically skipped
if llvmlite is not installed, so CI without LLVM still passes cleanly.
"""

import pytest

llvm = pytest.importorskip(
    "llvmlite.binding",
    reason="llvmlite not installed — skipping backend tests (need LLVM 14)"
)

from backend.codegen import generate_object_code

SAMPLES_DIR = __import__("os").path.join(__import__("os").path.dirname(__file__), "samples")


def pipeline_ir(c_filename: str) -> str:
    from frontend.lexer import lex
    from frontend.parser import parse
    from frontend.semantic import analyze
    from ir.generator import generate_ir
    with open(__import__("os").path.join(SAMPLES_DIR, c_filename)) as f:
        src = f.read()
    return generate_ir(analyze(parse(lex(src))))


class TestGenerateObjectCode:

    def test_returns_bytes(self):
        ir = pipeline_ir("01_return.c")
        result = generate_object_code(ir)
        assert isinstance(result, bytes)

    def test_non_empty_object_code(self):
        ir = pipeline_ir("02_add.c")
        result = generate_object_code(ir)
        assert len(result) > 0

    def test_all_samples_compile(self):
        for n in ["01_return", "02_add", "03_if_else", "04_while", "05_func_call"]:
            ir = pipeline_ir(f"{n}.c")
            obj = generate_object_code(ir)
            assert isinstance(obj, bytes) and len(obj) > 0, \
                f"{n}.c produced empty object code"

    def test_default_triple_works(self):
        """generate_object_code with no triple uses native target."""
        ir = pipeline_ir("01_return.c")
        obj = generate_object_code(ir, target_triple=None)
        assert len(obj) > 0

    def test_invalid_ir_raises(self):
        with pytest.raises(RuntimeError, match="LLVM IR verification failed|malformed|expected"):
            generate_object_code("this is not valid LLVM IR")

    def test_optimized_ir_compiles(self):
        from optimizer.passes import optimize
        ir = pipeline_ir("03_if_else.c")
        opt_ir = optimize(ir)
        obj = generate_object_code(opt_ir)
        assert len(obj) > 0
