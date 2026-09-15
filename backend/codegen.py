"""
Backend code generator for the MiniC compiler.

Compiles the optimized LLVM IR text string into native machine object code
using llvmlite's LLVM bindings.

Interface defined in docs/SPEC.md §4:
    generate_object_code(ir: str, target_triple: str) -> bytes

Requirements
------------
* llvmlite 0.40.0  (pip install llvmlite)
* LLVM 14 system libraries
    Ubuntu:  sudo apt-get install llvm-14 llvm-14-dev
    macOS:   brew install llvm@14

If llvmlite is not installed, the function raises a clear RuntimeError with
installation instructions rather than an obscure ImportError.
"""

from __future__ import annotations
import platform
from typing import Optional


def _get_native_triple() -> str:
    """Return the LLVM target triple for the current machine."""
    try:
        import llvmlite.binding as llvm  # type: ignore
        llvm.initialize()
        llvm.initialize_native_target()
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
            "Install it with:  pip install llvmlite==0.40.0\n"
            "You also need LLVM 14 system libraries:\n"
            "  Ubuntu: sudo apt-get install llvm-14 llvm-14-dev\n"
            "  macOS:  brew install llvm@14"
        )

    # ── Initialise LLVM targets ───────────────────────────────────────────────
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

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
