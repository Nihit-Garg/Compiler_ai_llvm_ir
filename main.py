#!/usr/bin/env python3
"""
MiniC Compiler — Verified LLM-Guided Optimization
===================================================

Full pipeline driver:
    source.c → lex → parse → semantic analysis → LLVM IR
             → AI advisor (Gemini) → optimizer passes
             → differential verifier → (fallback if unsafe)
             → optimized IR  [→ object code if --compile]

Usage
-----
    python main.py <source.c> [options]

Options
-------
    --output, -o <file>   Write optimized IR to this path (default: <name>.ll)
    --report, -r <file>   Write JSON report to this path   (default: <name>_report.json)
    --compile, -c         Also compile IR → object code    (requires llvmlite + LLVM 14)
    --no-opt              Skip the AI advisor; apply the default pass order directly
    --quiet, -q           Suppress banner and step output; print IR to stdout only

Environment Variables
---------------------
    GEMINI_API_KEY   Free Gemini key — https://aistudio.google.com/apikey
                     If unset, the advisor falls back to the fixed default pass order.
    ADVISOR_MODEL    Gemini model name (default: gemini-1.5-flash)

Example
-------
    export GEMINI_API_KEY="..."
    python main.py tests/samples/02_add.c
    python main.py tests/samples/04_while.c --compile
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# ── Ensure project root is on sys.path ───────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.lexer    import lex
from frontend.parser   import parse
from frontend.semantic import analyze
from ir.generator      import generate_ir
from verifier.pipeline import apply_optimizations
from optimizer.passes  import optimize as default_optimize


# ---------------------------------------------------------------------------
# Terminal colours (graceful fallback for Windows/dumb terminals)
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_DIM    = "\033[2m"


def _c(text: str, *codes: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _RESET


# ---------------------------------------------------------------------------
# Step printer
# ---------------------------------------------------------------------------

def _step(n: int, total: int, label: str, quiet: bool):
    if not quiet:
        print(f"  {_c(f'[{n}/{total}]', _DIM)} {label}", end="", flush=True)


def _ok(quiet: bool):
    if not quiet:
        print(f"  {_c('✓', _GREEN)}")


def _warn(msg: str, quiet: bool):
    if not quiet:
        print(f"  {_c('⚠', _YELLOW)}  {msg}")


def _fail(msg: str):
    print(f"\n  {_c('✗', _RED)}  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER = f"""
{_c('═' * 58, _CYAN)}
  {_c('MiniC Compiler', _BOLD)}  {_c('— Verified LLM-Guided Optimization', _DIM)}
{_c('═' * 58, _CYAN)}"""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compile_source(
    source_path: str,
    output_ir: str,
    output_report: str,
    do_compile: bool,
    no_opt: bool,
    quiet: bool,
) -> int:
    """Run the full compiler pipeline. Returns exit code (0 = success)."""

    src_path = Path(source_path)
    if not src_path.exists():
        _fail(f"File not found: {source_path}")
        return 1

    if not quiet:
        print(_BANNER)
        print(f"  {_c('Source:', _DIM)} {src_path}")
        print()

    t_start = time.perf_counter()

    # ── Read source ───────────────────────────────────────────────────────────
    with open(src_path) as f:
        source = f.read()

    TOTAL = 4 if not do_compile else 5

    # ── Step 1: Lexing & Parsing ──────────────────────────────────────────────
    _step(1, TOTAL, "Lexing & Parsing .................. ", quiet)
    try:
        tokens = lex(source)
        ast    = parse(tokens)
    except Exception as e:
        _fail(f"Parse error: {e}")
        return 1
    _ok(quiet)

    # ── Step 2: Semantic Analysis ─────────────────────────────────────────────
    _step(2, TOTAL, "Semantic Analysis ................. ", quiet)
    try:
        ast = analyze(ast)
    except Exception as e:
        _fail(f"Semantic error: {e}")
        return 1
    _ok(quiet)

    # ── Step 3: IR Generation ─────────────────────────────────────────────────
    _step(3, TOTAL, "IR Generation ..................... ", quiet)
    try:
        raw_ir = generate_ir(ast)
    except Exception as e:
        _fail(f"IR generation error: {e}")
        return 1
    _ok(quiet)

    # ── Step 4: Optimization ──────────────────────────────────────────────────
    if no_opt:
        _step(4, TOTAL, "Optimization (default passes) ..... ", quiet)
        try:
            final_ir = default_optimize(raw_ir)
        except Exception as e:
            _fail(f"Optimization error: {e}")
            return 1
        report = {
            "suggested_passes": ["constant_folding", "dead_code_elimination",
                                  "common_subexpression_elimination", "peephole"],
            "decision": "accepted",
            "rejection_reason": None,
            "fallback_passes": None,
            "final_passes": ["constant_folding", "dead_code_elimination",
                              "common_subexpression_elimination", "peephole"],
            "mode": "no-opt (default passes, AI advisor skipped)",
        }
        _ok(quiet)
    else:
        _step(4, TOTAL, "AI Optimization + Verification .... ", quiet)
        try:
            final_ir, report = apply_optimizations(raw_ir)
        except Exception as e:
            _fail(f"Optimization pipeline error: {e}")
            return 1

        if not quiet:
            decision = report.get("decision", "unknown")
            colour   = _GREEN if decision == "accepted" else _YELLOW
            print(f"  {_c('✓', _GREEN)}")
            print(f"       Suggested : {report.get('suggested_passes')}")
            print(f"       Decision  : {_c(decision.upper(), colour, _BOLD)}", end="")
            if decision == "rejected":
                print(f"  ({report.get('rejection_reason', '')})")
            else:
                print()
            print(f"       Applied   : {report.get('final_passes')}")

    # ── Write IR ──────────────────────────────────────────────────────────────
    ir_path = Path(output_ir)
    ir_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ir_path, "w") as f:
        f.write(final_ir)

    # ── Write report ──────────────────────────────────────────────────────────
    report["source_file"] = str(src_path)
    report["ir_output"]   = str(ir_path)
    report["elapsed_s"]   = round(time.perf_counter() - t_start, 3)

    rpt_path = Path(output_report)
    rpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Step 5 (optional): Compile to object code ─────────────────────────────
    obj_path = None
    if do_compile:
        _step(5, TOTAL, "Object Code Generation ............ ", quiet)
        try:
            from backend.codegen import generate_object_code
            obj_bytes = generate_object_code(final_ir)
            obj_path  = ir_path.with_suffix(".o")
            with open(obj_path, "wb") as f:
                f.write(obj_bytes)
            _ok(quiet)
        except RuntimeError as e:
            _warn(str(e).splitlines()[0], quiet)
        except Exception as e:
            _warn(f"Object code generation failed: {e}", quiet)

    # ── Summary ───────────────────────────────────────────────────────────────
    if not quiet:
        print()
        print(_c("  ── Output ──────────────────────────────────────────", _DIM))
        print(f"  IR   → {_c(str(ir_path), _CYAN)}")
        if obj_path and obj_path.exists():
            print(f"  OBJ  → {_c(str(obj_path), _CYAN)}")
        print(f"  JSON → {_c(str(rpt_path), _CYAN)}")
        elapsed = report["elapsed_s"]
        print(f"\n  {_c(f'Done in {elapsed}s', _DIM)}\n")
    else:
        # Quiet mode: print IR to stdout
        print(final_ir, end="")

    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="MiniC Compiler — Verified LLM-Guided Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Example")[1] if "Example" in __doc__ else "",
    )
    p.add_argument("source",            help="Path to the MiniC .c source file")
    p.add_argument("--output", "-o",    help="Write optimized IR here (default: <name>.ll)")
    p.add_argument("--report", "-r",    help="Write JSON report here (default: <name>_report.json)")
    p.add_argument("--compile", "-c",   action="store_true",
                   help="Also compile IR → object code (requires llvmlite + LLVM 14)")
    p.add_argument("--no-opt",          action="store_true",
                   help="Skip AI advisor; apply default pass order directly")
    p.add_argument("--quiet", "-q",     action="store_true",
                   help="Suppress step output; print optimized IR to stdout only")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    src   = Path(args.source)
    stem  = src.stem

    ir_out  = args.output  or str(src.parent / f"{stem}.ll")
    rpt_out = args.report  or str(src.parent / f"{stem}_report.json")

    return compile_source(
        source_path=str(src),
        output_ir=ir_out,
        output_report=rpt_out,
        do_compile=args.compile,
        no_opt=args.no_opt,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
