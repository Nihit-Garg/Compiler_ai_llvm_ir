# Verified LLM-Guided Compiler Optimization

## What This Project Is

This is a compiler for a constrained C-like language that generates LLVM IR and applies standard optimization passes — but instead of running those passes in one fixed, hardcoded order (like `-O2`/`-O3` always does), an LLM is asked to suggest which passes to apply and in what sequence for each specific program.

The key idea: **we never trust the LLM's suggestion blindly.** Every suggested optimization is verified for correctness before being accepted. If verification fails, the compiler falls back to a fixed default pass order, and that order is verified the same way. If it fails too, the IR is left unoptimized. This means the AI layer can only make the compiler *better*, never *incorrect* — the compiler's core guarantee (correct output) is never put at risk by the AI component.

In short: classic compiler + an adaptive optimization advisor + a safety net that keeps it honest.

## Why This Exists

Fixed-order optimizers (GCC, Clang) don't adapt per-program. ML-guided optimizers (e.g. AutoPhase, MLGO) do adapt, but offer no guarantee they won't occasionally suggest something that changes program behavior. This project closes that gap at a small scale — combining adaptive, LLM-driven decisions with a differential-testing verifier that guarantees safety.

## How It Operates — End to End

```
Source (.c subset)
   → Lexer → Parser → AST
   → Semantic Analysis (type check, scope, symbol table)
   → LLVM IR Generation
   → LLM Advisor: suggests an optimization pass sequence
   → Apply suggested passes → Optimized IR
   → Verifier: differential test optimized IR vs. original IR
        → if behaviorally equivalent → ACCEPT
        → if not → REJECT → apply default fixed pass order, verify it too
             → if verified → use it
             → if not → keep the unoptimized IR
   → Final IR check (structural + LLVM's own verifier)
   → Code Generation (optional) → Final Output + Log Report
```

The JSON report records where the suggestion came from (`advisor_source`: the LLM, or a fallback such as `no_api_key`), what was accepted or rejected, which passes were applied, and the instruction count before and after optimization (`instructions_before`, `instructions_after`). This becomes our evaluation data.

## Project Structure

- `/frontend` — lexer, parser, AST, semantic analysis
- `/ir` — AST → LLVM IR lowering
- `/optimizer` — individual optimization passes (constant folding, DCE, CSE, peephole)
- `/advisor` — LLM integration: sends IR + pass list, parses suggested sequence
- `/verifier` — differential testing harness, fallback logic
- `/backend` — final code generation
- `/tests` — unit and end-to-end tests; `tests/samples/` holds sample programs and their expected LLVM IR (`.ll`) as ground truth
- `/docs` — SPEC.md and design docs
- `/evaluation` — benchmark programs, the evaluation script and its results (see [Evaluation](#evaluation))

## The Source of Truth: `docs/SPEC.md`

Before writing any code, read `docs/SPEC.md`. It contains:
- The exact grammar (EBNF) for the supported language subset
- The exact IR format and worked examples (source → IR)
- Function signatures / interface contracts for every module
- What is explicitly **out of scope** (no pointers, arrays, structs, etc. unless stated)

This file is the single source of truth for the project. If your code and `SPEC.md` disagree, `SPEC.md` wins — raise it in the team channel before changing either.

## Working With LLMs on This Codebase (Read Before Generating Code)

Every contributor is expected to use an LLM to help write code. To avoid hallucinated APIs, invented features, or mismatched interfaces, follow this rule strictly:

**Always give the LLM three things before asking it to generate code:**
1. The relevant section of `docs/SPEC.md`
2. The existing stub/interface for the file you're working on (function signatures, docstrings — don't let the LLM invent its own)
3. At least one matching example from `/tests/samples/`

Never prompt an LLM with just "write me a parser" or "implement dead code elimination" with no context — that's when it invents behavior that doesn't match the rest of the system. Treat the LLM as a fast typist working strictly inside a spec you hand it, not as a design authority.

If the LLM's output doesn't match an existing interface in the codebase, don't change the interface to fit the LLM's output — go back and re-prompt with the correct interface attached.



## Toolchain & Environment

- **Python**: 3.10, 3.11, 3.12 or 3.13
- **llvmlite**: 0.40.0 or later. The llvmlite wheel bundles its own copy of LLVM, so you do not need to install LLVM separately.

Tested combinations:

| Python | llvmlite | Where |
|---|---|---|
| 3.10 | 0.40.0 | CI |
| 3.11 | 0.40.0 | local (Windows) |
| 3.11 | 0.50.0 | local (Windows), fresh clone |
| 3.12 | 0.49.0 | local (Windows) |
| 3.12 | 0.50.0 | local (Windows), fresh clone |
| 3.12 | latest release | CI |
| 3.13 | 0.50.0 | independent check by the project team |

llvmlite 0.40.0 has no wheels for Python 3.12 or later. On 3.12 and 3.13, `pip install -r requirements.txt` installs a current llvmlite.

## Getting Started

1. Clone the repository and create a virtual environment:
   ```bash
   git clone <repo-url>
   cd <repo>
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the tests:
   ```bash
   pytest tests/
   ```
4. Compile a program:
   ```bash
   python main.py tests/samples/02_add.c
   ```
   Output goes to a `build/` folder in the **current folder**: `build/02_add.ll` (the optimized IR) and `build/02_add_report.json` (the report). Add `--compile` to also write `build/02_add.o`. Use `-o` and `-r` to choose other paths. Two sources with the same file name (say `a/test.c` and `b/test.c`) write to the same `build/test.ll`, so the second run overwrites the first unless you give `-o`.

   Other options: `--no-advisor` skips the LLM and applies the default pass order, which is still verified. `--no-opt` is an old alias for it, and despite the name it does not turn optimization off. `--quiet` prints only the IR. Exit status is 0 on success and 1 on any error, including a failed final IR check or a failed `--compile`.

5. Optional: to use the LLM advisor, copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Without a key, the default pass order is used, and the report says so (`advisor_source: "no_api_key"`).

See `docs/SPEC.md` for the language grammar and detailed module contracts before contributing.

## Evaluation

The evaluation asks two questions: does the LLM's pass order beat the fixed default order, and does the safety net catch bad suggestions? Full tables are in [`evaluation/results.md`](evaluation/results.md).

```bash
python evaluation/run_eval.py          # rebuild the tables from saved LLM answers (no API key needed)
python evaluation/run_eval.py --live   # ask Gemini again (about a minute per program)
```

**Setup.** The programs are the 5 samples in `tests/samples/` plus 10 benchmarks in `evaluation/benchmarks/`, which are written so that pass order matters. For each program the script counts instructions after four options:
- no optimization;
- the default order (constant folding, DCE, CSE, peephole);
- the order the LLM (`gemini-3.5-flash-lite`) suggested, after the verifier accepted it;
- the best order, found by trying all 64 orderings.

**Result: the LLM's order is useful.**

| | Instructions (15 programs) | Removed |
|---|---:|---:|
| No optimization | 173 | — |
| Default order | 163 | 10 |
| LLM order | **130** | **43** |
| Best possible | 130 | 43 |

- The LLM beat the default order on 9 of the 15 programs and matched it on the other 6. It was never worse.
- It found the best possible order for all 15.
- The default order is weak because it runs DCE before peephole. That leaves behind the instructions peephole replaces.
- **The best order depends on the program, and the LLM adapted to it.** Constant folding and peephole can need opposite orders:
  - `b3` (`a * (3 - 2)`) must fold first: `3 - 2` becomes `1`, then `a * 1` simplifies to `a`.
  - `b8`–`b10` (for example `(a * 0 + 2) * 3`) must simplify first: `a * 0 + 2` becomes `2`, then `2 * 3` folds to `6`.

  Each pass runs once, so no single fixed order is best for both kinds. The LLM suggested CF → PH → DCE for `b3` and PH → CF → DCE for `b8`–`b10`. These four benchmarks were written to show this conflict.
- Every LLM suggestion was checked by the verifier before use.

**What made the difference: describing the passes.** The first run gave the LLM only the pass names (results in [`evaluation/results_names_only.md`](evaluation/results_names_only.md)). That run covered the first 12 programs. It beat the default on 3 of them and reached the best result on 8 of the 9 it answered. On the same 12 programs, adding a one-line description of each pass to the prompt, plus the goal (fewest instructions), raised this to 6, and the best result on all 12. The names-only baseline does not include `b8`–`b10`, which were added later.

**Result: the safety net works.** The script runs the real pipeline on four cases:

| Case | What happened |
|---|---|
| The LLM suggests a faulty pass (turns `sub` into `add`) | The verifier rejects it and names the input that shows the bug: `@diff(-7, -7): original returned 0, optimized returned -14`. The default order is verified and used instead. |
| A program the verifier cannot check (never terminates) | The suggestion and the fallback are both rejected, and the IR is left unchanged. |
| The LLM's answer is unusable | Reported as `unusable_response`. The default order is used, after it is verified. |
| The LLM answers `[]` | No passes are applied, and the IR is left unchanged. |

In every case, the IR that comes out is equivalent to the IR that went in.

**Limits.**
- The verifier uses differential testing on sample inputs, not a proof.
- Instruction count is the only measure; runtime is not timed.
- The benchmarks are small programs written for this evaluation.
- Live answers took 30–100 seconds each, and a few calls failed with API errors. Those programs fell back to the default order and were asked again with `--retry-errors`.
