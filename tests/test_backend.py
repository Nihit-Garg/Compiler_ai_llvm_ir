"""
Tests for backend/codegen.py.

The llvmlite wheel bundles LLVM, so no system LLVM is needed. Tests are
skipped automatically if llvmlite is not installed.
"""

import pytest

llvm = pytest.importorskip(
    "llvmlite.binding",
    reason="llvmlite not installed — skipping backend tests"
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

    def test_works_when_initialize_is_deprecated(self, monkeypatch):
        """
        Newer llvmlite initialises LLVM itself and makes initialize() raise
        RuntimeError.  The backend must still compile on such a version.
        """
        def deprecated():
            raise RuntimeError("llvmlite.binding.initialize() is deprecated "
                               "and will be removed.")
        monkeypatch.setattr(llvm, "initialize", deprecated)
        assert len(generate_object_code(pipeline_ir("02_add.c"))) > 0

    def test_initialize_is_still_called_where_it_works(self, monkeypatch):
        """Older llvmlite (0.40.x) needs initialize(); it must not be skipped."""
        calls = []
        monkeypatch.setattr(llvm, "initialize", lambda: calls.append(1))
        generate_object_code(pipeline_ir("01_return.c"))
        assert calls == [1]

    def test_any_pass_order_gives_ir_that_llvm_accepts(self):
        """
        Every ordering of the four passes must leave IR that real LLVM parses.
        (Deleting an unnamed temporary used to leave a numbering gap.)
        """
        from itertools import permutations
        from frontend.lexer import lex
        from frontend.parser import parse
        from frontend.semantic import analyze
        from ir.generator import generate_ir
        from optimizer import passes

        src = "int f(int a, int b) { int r = (0 * a) + (b + b) * (a * 1) + 2 * 3; return r; }"
        ir = generate_ir(analyze(parse(lex(src))))
        names = ["constant_folding", "dead_code_elimination",
                 "common_subexpression_elimination", "peephole"]
        for size in range(1, len(names) + 1):
            for order in permutations(names, size):
                out = ir
                for name in order:
                    out = getattr(passes, name)(out)
                obj = generate_object_code(out)      # raises if LLVM rejects it
                assert len(obj) > 0, f"empty object code for order {order}"


# ---------------------------------------------------------------------------
# Programs beyond the five samples: real LLVM must accept what we generate
# ---------------------------------------------------------------------------

from tests.test_ir_generator import PROGRAMS_BEYOND_SAMPLES


def _compile(source: str) -> str:
    from frontend.lexer import lex
    from frontend.parser import parse
    from frontend.semantic import analyze
    from ir.generator import generate_ir
    return generate_ir(analyze(parse(lex(source))))


def _run_in_llvm(ir: str, fn: str, args):
    """JIT-compile the IR with LLVM and call one function."""
    import ctypes
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    module = llvm.parse_assembly(ir)
    module.verify()
    machine = llvm.Target.from_default_triple().create_target_machine()
    engine = llvm.create_mcjit_compiler(module, machine)
    engine.finalize_object()
    proto = ctypes.CFUNCTYPE(ctypes.c_int32, *([ctypes.c_int32] * len(args)))
    return proto(engine.get_function_address(fn))(*args)


@pytest.mark.parametrize("name, source, fn, cases", PROGRAMS_BEYOND_SAMPLES,
                         ids=[p[0] for p in PROGRAMS_BEYOND_SAMPLES])
class TestProgramsBeyondSamples:

    def test_unoptimized_ir_is_accepted_by_llvm(self, name, source, fn, cases):
        assert len(generate_object_code(_compile(source))) > 0

    def test_optimized_ir_is_accepted_by_llvm(self, name, source, fn, cases):
        from optimizer.passes import optimize
        assert len(generate_object_code(optimize(_compile(source)))) > 0

    def test_llvm_computes_the_c_result(self, name, source, fn, cases):
        from optimizer.passes import optimize
        raw = _compile(source)
        for ir in (raw, optimize(raw)):
            for args, expected in cases:
                assert _run_in_llvm(ir, fn, args) == expected, f"{name}{args}"
