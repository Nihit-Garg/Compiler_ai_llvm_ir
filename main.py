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
    python main.py --interactive [options]     (type the program, end with Ctrl-D)

Options
-------
    --output, -o <file>   Write optimized IR to this path (default: build/<name>.ll)
    --report, -r <file>   Write JSON report to this path   (default: build/<name>_report.json)
    --compile, -c         Also compile IR → object code    (requires llvmlite)
    --no-advisor          Do not ask the LLM; use the default pass order (still verified).
                          --no-opt is an older alias: it does NOT turn optimization off.
    --quiet, -q           Suppress banner and step output; print IR to stdout only
    --show-steps, -s      Print the output of every stage: tokens, AST, symbols,
                          raw IR, the IR after each pass, the verifier's test
                          runs and the final check
    --interactive, -i     Read the program from the terminal (stdin) instead of
                          a file; "-" as the source does the same.  Output goes
                          to build/input.ll unless -o says otherwise

The default build/ folder is created in the current working directory, so the
source folder (and the expected-output files in tests/samples/) is never
written to unless -o or -r points there.

Exit status
-----------
    0   IR written, and it passed the final check (verify_ir, plus LLVM's own
        parser and verifier when llvmlite is installed); with --compile, the
        object file was written too
    1   any error: unreadable source, compile error, a final IR check that
        failed, or --compile that could not produce an object file.  The
        report still records what happened whenever the pipeline got that far.

Environment Variables
---------------------
    GEMINI_API_KEY   Free Gemini key — https://aistudio.google.com/apikey
                     If unset, the advisor falls back to the fixed default pass order.
    ADVISOR_MODEL    Gemini model name (default: gemini-3.5-flash-lite)

Example
-------
    export GEMINI_API_KEY="..."
    python main.py tests/samples/02_add.c
    python main.py tests/samples/04_while.c --compile
    python main.py evaluation/benchmarks/b8_simplify_then_fold.c --show-steps
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

# ── Ensure project root is on sys.path ───────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.lexer    import lex
from frontend.parser   import parse
from frontend.semantic import analyze
from ir.generator      import generate_ir
from verifier.pipeline import apply_optimizations
from verifier.check    import verify_ir, llvm_check, LLVM_FAILED
import show_steps as steps

DEFAULT_BUILD_DIR = "build"
STDIN_SOURCE = "<stdin>"     # source_file in the report for --interactive


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
    show: bool = False,
    source_text: Optional[str] = None,
) -> int:
    """
    Run the full compiler pipeline. Returns exit code (0 = success).

    show:        print every stage's output (--show-steps); ignored with quiet.
    source_text: the program itself, when it was typed in rather than read
                 from source_path (--interactive).
    """
    show = show and not quiet

    src_path = Path(source_path)
    if source_text is None and not src_path.exists():
        _fail(f"File not found: {source_path}")
        return 1

    if not quiet:
        print(_BANNER)
        print(f"  {_c('Source:', _DIM)} {src_path}")
        print()

    t_start = time.perf_counter()

    # ── Read source ───────────────────────────────────────────────────────────
    if source_text is not None:
        source = source_text
    else:
        with open(src_path, encoding="utf-8") as f:
            source = f.read()
    if show:
        steps.show_source(source)
        print()

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
    if show:
        steps.show_tokens(tokens)
        steps.show_ast(ast)
        print()

    # ── Step 2: Semantic Analysis ─────────────────────────────────────────────
    _step(2, TOTAL, "Semantic Analysis ................. ", quiet)
    try:
        ast = analyze(ast)
    except Exception as e:
        _fail(f"Semantic error: {e}")
        return 1
    _ok(quiet)
    if show:
        steps.show_symbols(ast)
        print()

    # ── Step 3: IR Generation ─────────────────────────────────────────────────
    _step(3, TOTAL, "IR Generation ..................... ", quiet)
    try:
        raw_ir = generate_ir(ast)
    except Exception as e:
        _fail(f"IR generation error: {e}")
        return 1
    _ok(quiet)
    if show:
        steps.show_ir("Step 3: Generated LLVM IR (unoptimized)", raw_ir)
        print()

    # ── Step 4: Optimization ──────────────────────────────────────────────────
    # Both modes go through the verified pipeline; --no-advisor (no_opt) only
    # skips the LLM advisor and uses the default pass order as the suggestion.
    if no_opt:
        _step(4, TOTAL, "Default passes + Verification ..... ", quiet)
    else:
        _step(4, TOTAL, "AI Optimization + Verification .... ", quiet)
    try:
        final_ir, report = apply_optimizations(raw_ir, use_advisor=not no_opt)
    except Exception as e:
        _fail(f"Optimization pipeline error: {e}")
        return 1
    if no_opt:
        report["mode"] = "no-advisor (default passes, AI advisor skipped)"

    if not quiet:
        decision = report.get("decision", "unknown")
        colour   = _GREEN if decision == "accepted" else _YELLOW
        print(f"  {_c('✓', _GREEN)}")
        source = report.get("advisor_source")
        model  = report.get("advisor_model")
        print(f"       Advisor   : {source}" + (f" ({model})" if model else ""))
        print(f"       Suggested : {report.get('suggested_passes')}")
        print(f"       Decision  : {_c(decision.upper(), colour, _BOLD)}", end="")
        if decision == "rejected":
            print(f"  ({report.get('rejection_reason', '')})")
        else:
            print()
        fallback_decision = report.get("fallback_decision")
        if fallback_decision == "accepted":
            print(f"       Fallback  : {_c('ACCEPTED', _GREEN, _BOLD)}"
                  f"  (default pass order verified)")
        elif fallback_decision == "rejected":
            print(f"       Fallback  : {_c('REJECTED', _YELLOW, _BOLD)}"
                  f"  ({report.get('fallback_rejection_reason', '')})")
            print("                   IR left unoptimized")
        print(f"       Applied   : {report.get('final_passes')}")
        print(f"       Size      : {report.get('instructions_before')} → "
              f"{report.get('instructions_after')} instructions")
    if show:
        steps.show_optimization(raw_ir, report)

    exit_code = 0

    # ── Final IR check ────────────────────────────────────────────────────────
    # The verifier compared behaviour; this checks that what we are about to
    # hand out is IR at all — our own structural check, and LLVM's parser and
    # verifier whenever llvmlite is installed.
    structural_ok = verify_ir(final_ir)
    llvm_status, llvm_error = llvm_check(final_ir)
    final_ok = structural_ok and llvm_status != LLVM_FAILED
    report["final_ir_check"] = {
        "verify_ir":  structural_ok,
        "llvm":       llvm_status,
        "llvm_error": llvm_error,
        "passed":     final_ok,
    }
    if not final_ok:
        problem = llvm_error if llvm_status == LLVM_FAILED else "verify_ir() rejected it"
        _fail(f"Final IR check failed: {problem}")
        exit_code = 1
    if show:
        steps.show_final_check(report["final_ir_check"])
        steps.show_ir("Final optimized IR", final_ir)
        print()

    # ── Write IR ──────────────────────────────────────────────────────────────
    # Written even when the final check failed, so it can be inspected; the
    # report and the exit status say it is not valid.
    ir_path = Path(output_ir)
    ir_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ir_path, "w", encoding="utf-8") as f:
        f.write(final_ir)

    # ── Step 5 (optional): Compile to object code ─────────────────────────────
    obj_path = None
    if do_compile:
        report["object_output"] = None
        report["compile_error"] = None
        _step(5, TOTAL, "Object Code Generation ............ ", quiet)
        if not final_ok:
            report["compile_error"] = "skipped: final IR check failed"
            _warn(report["compile_error"], quiet)
        else:
            try:
                from backend.codegen import generate_object_code
                obj_bytes = generate_object_code(final_ir)
                obj_path  = ir_path.with_suffix(".o")
                with open(obj_path, "wb") as f:
                    f.write(obj_bytes)
                report["object_output"] = str(obj_path)
                _ok(quiet)
                if show:
                    steps.show_object(obj_path, obj_bytes)
            except Exception as e:
                report["compile_error"] = f"{type(e).__name__}: {e}"
                obj_path = None
        if report["compile_error"] is not None:
            if final_ok:
                _fail(f"Object code generation failed: {str(report['compile_error']).splitlines()[0]}")
            exit_code = 1

    # ── Write report ──────────────────────────────────────────────────────────
    report["source_file"] = STDIN_SOURCE if source_text is not None else str(src_path)
    report["ir_output"]   = str(ir_path)
    report["exit_code"]   = exit_code
    report["elapsed_s"]   = round(time.perf_counter() - t_start, 3)

    rpt_path = Path(output_report)
    rpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    if not quiet:
        print()
        print(_c("  ── Output ──────────────────────────────────────────", _DIM))
        print(f"  IR   → {_c(str(ir_path), _CYAN)}")
        if obj_path and obj_path.exists():
            print(f"  OBJ  → {_c(str(obj_path), _CYAN)}")
        print(f"  JSON → {_c(str(rpt_path), _CYAN)}")
        elapsed = report["elapsed_s"]
        status = "Done" if exit_code == 0 else "Finished with errors"
        print(f"\n  {_c(f'{status} in {elapsed}s', _DIM)}\n")
    else:
        # Quiet mode: print IR to stdout
        print(final_ir, end="")

    return exit_code


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
    p.add_argument("source", nargs="?",
                   help="Path to the MiniC .c source file (\"-\" reads it from stdin)")
    p.add_argument("--output", "-o",    help="Write optimized IR here (default: build/<name>.ll)")
    p.add_argument("--report", "-r",    help="Write JSON report here (default: build/<name>_report.json)")
    p.add_argument("--compile", "-c",   action="store_true",
                   help="Also compile IR → object code (requires llvmlite)")
    p.add_argument("--no-advisor", "--no-opt", dest="no_advisor", action="store_true",
                   help="Do not ask the LLM advisor; apply the default pass order "
                        "(still verified). --no-opt is an alias kept for old "
                        "scripts; it does not disable optimization")
    p.add_argument("--quiet", "-q",     action="store_true",
                   help="Suppress step output; print optimized IR to stdout only")
    p.add_argument("--show-steps", "-s", dest="show_steps", action="store_true",
                   help="Print the output of every stage: tokens, AST, symbols, "
                        "raw IR, the IR after each pass, the verifier's test runs "
                        "and the final check")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="Type the program in the terminal instead of giving a "
                        "file; end it with Ctrl-D (Ctrl-Z then Enter on Windows)")
    return p


def _read_program_from_stdin() -> str:
    """Read a MiniC program typed in the terminal (or piped in)."""
    if sys.stdin.isatty():
        print("Type your MiniC program. Finish with Ctrl-D "
              "(Ctrl-Z then Enter on Windows):", file=sys.stderr)
    return sys.stdin.read()


def _make_output_encoding_safe() -> None:
    """
    Keep printing from crashing on streams that cannot encode the banner's
    box-drawing characters and ✓ / ⚠ / ✗ (for example output piped or
    redirected on Windows, which uses cp1252).  Such characters are replaced
    with "?"; the stream's encoding is left alone, and the IR itself is ASCII.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def main() -> int:
    _make_output_encoding_safe()
    parser = _build_parser()
    args = parser.parse_args()

    from_stdin = args.interactive or args.source == "-"
    if not from_stdin and not args.source:
        parser.error("give a source file, or use --interactive to type the program")
    source_text = _read_program_from_stdin() if from_stdin else None

    src   = Path(STDIN_SOURCE if from_stdin else args.source)
    stem  = "input" if from_stdin else src.stem
    build = Path(DEFAULT_BUILD_DIR)

    ir_out  = args.output  or str(build / f"{stem}.ll")
    rpt_out = args.report  or str(build / f"{stem}_report.json")

    return compile_source(
        source_path=str(src),
        output_ir=ir_out,
        output_report=rpt_out,
        do_compile=args.compile,
        no_opt=args.no_advisor,
        quiet=args.quiet,
        show=args.show_steps,
        source_text=source_text,
    )


if __name__ == "__main__":
    sys.exit(main())
