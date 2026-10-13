"""
Backend code generator for the MiniC compiler.

Compiles the optimized LLVM IR text string into native machine object code
using llvmlite's LLVM bindings.

Interface defined in docs/SPEC.md §4:
    generate_object_code(ir: str, target_triple: str) -> bytes

Requirements
------------
* llvmlite >= 0.40.0  (pip install -r requirements.txt)
  The llvmlite wheel bundles its own copy of LLVM; no system LLVM is needed.
  Tested with llvmlite 0.40.0, 0.49.0 and 0.50.0.

If llvmlite is not installed, the function raises a clear RuntimeError with
installation instructions rather than an obscure ImportError.
"""

from __future__ import annotations
import platform
from typing import Optional


def _initialize_llvm(llvm) -> None:
    """
    Initialise the LLVM core and the native target.

    llvmlite up to 0.40.x needs an explicit llvm.initialize() call.  Newer
    releases initialise the core themselves and make initialize() raise a
    RuntimeError saying it is deprecated; only that error is ignored.
    """
    try:
        llvm.initialize()
    except RuntimeError as e:
        if "deprecated" not in str(e):
            raise
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()


def _get_native_triple() -> str:
    """Return the LLVM target triple for the current machine."""
    try:
        import llvmlite.binding as llvm  # type: ignore
        _initialize_llvm(llvm)
        return llvm.get_default_triple()
    except ImportError:
        # Fallback — construct a best-effort triple without llvmlite
        machine = platform.machine().lower()
        arch = "x86_64" if machine in ("x86_64", "amd64") else machine
        system = platform.system().lower()
        if system == "darwin":
            return f"{arch}-apple-macosx"
        elif system == "linux":
            return f"{arch}-pc-linux-gnu"
        else:
            return f"{arch}-pc-windows-msvc"


def generate_object_code(ir: str, target_triple: Optional[str] = None) -> bytes:
    """
    Compiles the LLVM IR string into native machine object code (.o format).

    Args:
        ir (str): The validated and optimized LLVM IR text.
        target_triple (str | None): LLVM target triple, e.g. 'x86_64-pc-linux-gnu'.
                                    If None, defaults to the native host triple.

    Returns:
        bytes: The generated ELF/Mach-O object code, ready to be linked.

    Raises:
        RuntimeError: If llvmlite is not installed.
        RuntimeError: If the IR is malformed or compilation fails.
    """
    # ── Import llvmlite ───────────────────────────────────────────────────────
    try:
        import llvmlite.binding as llvm  # type: ignore
    except ImportError:
        raise RuntimeError(
            "llvmlite is required for object code generation but is not installed.\n"
            "Install it with:  pip install -r requirements.txt"
        )

    # ── Initialise LLVM targets ───────────────────────────────────────────────
    _initialize_llvm(llvm)

    # ── Parse and verify the IR ───────────────────────────────────────────────
    try:
        mod = llvm.parse_assembly(ir)
        mod.verify()
    except Exception as e:
        raise RuntimeError(f"LLVM IR verification failed: {e}") from e

    # ── Select target ─────────────────────────────────────────────────────────
    triple = target_triple or llvm.get_default_triple()

    try:
        target = llvm.Target.from_triple(triple)
    except Exception as e:
        raise RuntimeError(f"Unknown target triple '{triple}': {e}") from e

    target_machine = target.create_target_machine(
        codemodel="default",
        reloc="pic",    # position-independent — needed for shared libs / Linux
    )

    # ── Emit object code ──────────────────────────────────────────────────────
    try:
        obj_bytes: bytes = target_machine.emit_object(mod)
    except Exception as e:
        raise RuntimeError(f"Object code emission failed: {e}") from e

    return obj_bytes
