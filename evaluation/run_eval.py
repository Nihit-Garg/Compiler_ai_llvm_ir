#!/usr/bin/env python3
"""
Evaluation for the MiniC compiler: is the LLM advisor useful, and does the
safety net work?

Part 1 — Usefulness.  For every program in tests/samples/ and
evaluation/benchmarks/ it compares the instruction count of:
    unoptimized IR | default pass order | LLM-suggested order | best order
"best" is found by trying all 64 orderings of the four passes (each used at
most once) and keeping the smallest result that the verifier accepts.

Part 2 — Safety net.  Runs the real pipeline (apply_optimizations) on four
situations and shows what it decided:
    a. the LLM suggests a faulty pass       -> rejected, verified fallback used
    b. a program the verifier cannot check  -> both rejected, IR left unchanged
    c. the LLM answer is unusable           -> default order used, verified
    d. the LLM answers []                   -> no passes, IR unchanged

Usage
-----
    python evaluation/run_eval.py                 # replay saved LLM answers (no key needed)
    python evaluation/run_eval.py --live          # ask Gemini again for every program
    python evaluation/run_eval.py --retry-errors  # ask again only where the API call failed

Outputs evaluation/results.md and evaluation/results.json.  LLM answers are
saved in evaluation/llm_suggestions.json, so the tables can be rebuilt
without an API key.  A live run takes about a minute per program with
gemini-3.5-flash-lite.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.lexer    import lex
from frontend.parser   import parse
from frontend.semantic import analyze
from ir.generator      import generate_ir
from advisor.heuristics import _VALID_PASSES
from verifier.check    import count_instructions, verify_equivalence
from verifier.pipeline import apply_optimizations, _apply_pass_sequence
import verifier.pipeline as pipeline

HERE        = Path(__file__).resolve().parent
def _by_number(path: Path):
    """b2_x before b10_y: sort on the length of the prefix, then the name."""
    prefix = path.stem.split("_")[0]
    return (len(prefix), path.stem)


PROGRAMS    = sorted((ROOT / "tests" / "samples").glob("*.c")) + \
              sorted((HERE / "benchmarks").glob("*.c"), key=_by_number)


def compile_c(source: str) -> str:
    return generate_ir(analyze(parse(lex(source))))


def short(passes) -> str:
    """constant_folding -> CF etc., for compact tables."""
    abbrev = {"constant_folding": "CF", "dead_code_elimination": "DCE",
              "common_subexpression_elimination": "CSE", "peephole": "PH"}
    return " → ".join(abbrev.get(p, p) for p in passes) if passes else "(none)"


# ---------------------------------------------------------------------------
# Part 1: usefulness
# ---------------------------------------------------------------------------

def best_order(ir: str):
    """Smallest verified result over all orderings; ties go to the shorter order."""
    best = (count_instructions(ir), [])
    for k in range(1, len(_VALID_PASSES) + 1):
        for order in itertools.permutations(_VALID_PASSES, k):
            out = _apply_pass_sequence(ir, list(order))
            n = count_instructions(out)
            if n < best[0] and verify_equivalence(ir, out):
                best = (n, list(order))
    return best


def replay(answer):
    """Make the advisor return a saved answer instead of calling the API."""
    return patch.object(pipeline, "get_advice_with_source",
                        return_value=(answer["suggested"],
                                      {"source": answer["source"], "model": answer["model"]}))


def evaluate_program(path: Path, saved: dict, live: bool, retry_errors: bool = False) -> dict:
    ir = compile_c(path.read_text(encoding="utf-8"))
    name = path.stem

    default_ir, default_report = apply_optimizations(ir, use_advisor=False)

    failed_before = saved.get(name, {}).get("source") not in (None, "llm")
    if live or name not in saved or (retry_errors and failed_before):
        t0 = time.perf_counter()
        llm_ir, report = apply_optimizations(ir, use_advisor=True)
        saved[name] = {"suggested": report["suggested_passes"],
                       "source": report["advisor_source"],
                       "model": report["advisor_model"],
                       "seconds": round(time.perf_counter() - t0, 1)}
    else:
        with replay(saved[name]):
            llm_ir, report = apply_optimizations(ir, use_advisor=True)

    best_n, best_passes = best_order(ir)
    return {
        "program": name,
        "unoptimized": count_instructions(ir),
        "default": count_instructions(default_ir),
        "default_decision": default_report["decision"],
        "llm": count_instructions(llm_ir),
        "llm_source": report["advisor_source"],
        "llm_suggested": report["suggested_passes"],
        "llm_decision": report["decision"],
        "llm_final_passes": report["final_passes"],
        "best": best_n,
        "best_passes": best_passes,
    }


# ---------------------------------------------------------------------------
# Part 2: safety net
# ---------------------------------------------------------------------------

def broken_pass(ir: str) -> str:
    """A deliberately wrong 'optimization': it turns every subtraction into an addition."""
    return ir.replace(" sub i32 ", " add i32 ")


def fake_llm(passes, source="llm"):
    return patch.object(pipeline, "get_advice_with_source",
                        return_value=(passes, {"source": source, "model": "simulated"}))


def safety_cases() -> list:
    cases = []

    def run(title, expectation, src, *patches):
        ir = compile_c(src)
        with patch.dict(pipeline._PASS_FN, {"broken_pass": broken_pass}):
            for p in patches:
                p.start()
            try:
                final_ir, r = apply_optimizations(ir, use_advisor=True)
            finally:
                for p in reversed(patches):
                    p.stop()
        cases.append({
            "case": title,
            "expected": expectation,
            "suggested": r["suggested_passes"],
            "advisor_source": r["advisor_source"],
            "decision": r["decision"],
            "rejection_reason": r["rejection_reason"],
            "fallback_decision": r["fallback_decision"],
            "fallback_rejection_reason": r["fallback_rejection_reason"],
            "final_passes": r["final_passes"],
            "ir_unchanged": final_ir == ir,
            "final_equivalent": final_ir == ir or verify_equivalence(ir, final_ir),
        })

    run("a. LLM suggests a faulty pass (sub turned into add)",
        "suggestion rejected; default order verified and used",
        "int diff(int a, int b) { return a - b + 0; }",
        fake_llm(["peephole", "broken_pass"]))

    run("b. Program the verifier cannot check (never terminates)",
        "suggestion and fallback both rejected; IR left unchanged",
        "int spin() { int i = 0; while (i == 0) { i = 0 * 5; } return i; }",
        fake_llm(["peephole", "dead_code_elimination"]))

    run("c. LLM answer is unusable",
        "advisor falls back to the default order, which is verified",
        "int f(int a) { return a * 1 + 0; }",
        patch.dict("os.environ", {"GEMINI_API_KEY": "simulated"}),
        patch("advisor.heuristics._call_gemini", return_value="I think you should optimize it."))

    run("d. LLM answers [] (no optimization)",
        "no passes applied; IR unchanged",
        "int f(int a) { return a * 1 + 0; }",
        fake_llm([]))
    return cases


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def to_markdown(rows, cases, model) -> str:
    out = ["# Evaluation results", "",
           "Generated by `python evaluation/run_eval.py`. Instruction counts come from "
           "`verifier.check.count_instructions` (labels and comments are not counted). "
           f"LLM: `{model}`.", "",
           "## 1. Is the LLM's pass order useful?", "",
           "| Program | Unoptimized | Default order | LLM order | Best possible | Suggested order | Source | Verifier |",
           "|---|---:|---:|---:|---:|---|---|---|"]
    for r in rows:
        mark = " ✅" if r["llm"] < r["default"] else (" ❌" if r["llm"] > r["default"] else "")
        out.append(f"| {r['program']} | {r['unoptimized']} | {r['default']} | "
                   f"**{r['llm']}**{mark} | {r['best']} | {short(r['llm_suggested'])} | "
                   f"{r['llm_source']} | {r['llm_decision']} |")
    total = {k: sum(r[k] for r in rows) for k in ("unoptimized", "default", "llm", "best")}
    out.append(f"| **Total** | {total['unoptimized']} | {total['default']} | "
               f"**{total['llm']}** | {total['best']} | | | |")
    answered = [r for r in rows if r["llm_source"] == "llm"]
    better = sum(r["llm"] < r["default"] for r in answered)
    worse = sum(r["llm"] > r["default"] for r in answered)
    same = len(answered) - better - worse
    at_best = sum(r["llm"] == r["best"] for r in answered)
    removed_default = total["unoptimized"] - total["default"]
    removed_llm = total["unoptimized"] - total["llm"]
    out += ["",
            f"- The LLM answered for {len(answered)} of {len(rows)} programs. "
            "Where the source is not `llm`, the advisor fell back to the default order.",
            f"- Where it answered, its order beat the default order on **{better}** programs, "
            f"matched it on {same}, and was worse on {worse}.",
            f"- It reached the best possible result on {at_best} of those {len(answered)} programs.",
            f"- Instructions removed in total: default order {removed_default}, "
            f"LLM order {removed_llm}, best possible {total['unoptimized'] - total['best']}.",
            "- Every LLM order shown was checked by the differential verifier before use.",
            "",
            "## 2. Does the safety net work?", "",
            "| Case | Expected | LLM suggested | Decision | Fallback | Passes applied | Final IR correct |",
            "|---|---|---|---|---|---|---|"]
    for c in cases:
        fb = c["fallback_decision"] or "not needed"
        why = c["rejection_reason"] or ""
        if why:
            fb += f" (suggestion rejected: {why})"
        if c["fallback_rejection_reason"]:
            fb += f"; fallback rejected: {c['fallback_rejection_reason']}"
        out.append(f"| {c['case']} | {c['expected']} | {short(c['suggested'])} "
                   f"[{c['advisor_source']}] | {c['decision']} | {fb} | "
                   f"{short(c['final_passes'])} | {'yes' if c['final_equivalent'] else 'NO'}"
                   f"{' (unchanged)' if c['ir_unchanged'] else ''} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--live", action="store_true",
                    help="ask Gemini again (needs GEMINI_API_KEY) and save its answers")
    ap.add_argument("--retry-errors", action="store_true",
                    help="ask Gemini again only for programs whose saved answer was an error")
    ap.add_argument("--tag", default="",
                    help="use llm_suggestions_<tag>.json and write results_<tag>.* "
                         "(names_only holds the first run, with pass names only in the prompt)")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    suggestions = HERE / f"llm_suggestions{suffix}.json"

    saved = json.loads(suggestions.read_text(encoding="utf-8")) if suggestions.exists() else {}
    rows = []
    for path in PROGRAMS:
        if args.tag and not args.live and path.stem not in saved:
            # A tagged run is a frozen earlier experiment: asking the LLM now
            # would use today's prompt and file the answer under the old tag.
            print(f"  {path.stem}: no saved answer for '{args.tag}', skipped", flush=True)
            continue
        print(f"  {path.stem} ...", flush=True)
        rows.append(evaluate_program(path, saved, args.live, args.retry_errors))
        # Save after every program: a live run is slow and may be interrupted.
        suggestions.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")

    cases = safety_cases()
    models = sorted({str(v.get("model")) for v in saved.values()})
    (HERE / f"results{suffix}.json").write_text(
        json.dumps({"programs": rows, "safety_net": cases}, indent=2) + "\n", encoding="utf-8")
    (HERE / f"results{suffix}.md").write_text(to_markdown(rows, cases, ", ".join(models)),
                                              encoding="utf-8")
    print(f"  wrote {HERE / f'results{suffix}.md'}")

    ok = all(c["final_equivalent"] for c in cases)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
