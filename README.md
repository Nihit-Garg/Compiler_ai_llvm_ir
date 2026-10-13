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
