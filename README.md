# Verified LLM-Guided Compiler Optimization

## What This Project Is

This is a compiler for a constrained C-like language that generates LLVM IR and applies standard optimization passes — but instead of running those passes in one fixed, hardcoded order (like `-O2`/`-O3` always does), an LLM is asked to suggest which passes to apply and in what sequence for each specific program.

The key idea: **we never trust the LLM's suggestion blindly.** Every suggested optimization is verified for correctness before being accepted. If verification fails, the compiler silently falls back to a safe, fixed default pass order. This means the AI layer can only make the compiler *better*, never *incorrect* — the compiler's core guarantee (correct output) is never put at risk by the AI component.

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
        → if not → REJECT → apply default fixed pass order instead
   → Code Generation → Final Output + Log Report
```

The log report records what was suggested, what was accepted/rejected, and the resulting performance difference — this becomes our evaluation data.

## Project Structure

- `/frontend` — lexer, parser, AST, semantic analysis
- `/ir` — AST → LLVM IR lowering
- `/optimizer` — individual optimization passes (constant folding, DCE, CSE, peephole)
- `/advisor` — LLM integration: sends IR + pass list, parses suggested sequence
- `/verifier` — differential testing harness, fallback logic
- `/backend` — final code generation
- `/tests` — sample programs + expected AST/IR output (ground truth)
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

To ensure compatibility, please use the following pinned toolchain versions:
- **Python**: `3.10` or higher
- **LLVM**: `14.0.x` (or `14.x` series)
- **llvmlite**: `0.40.0` (must match the LLVM 14 version)

## Getting Started

1. Ensure LLVM 14 is installed on your system.
2. Clone the repository and set up the environment:
   ```bash
   git clone <repo-url>
   cd <repo>
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install llvmlite pytest
   # OR if requirements.txt exists:
   # pip install -r requirements.txt
   ```
4. Run tests to verify the setup:
   ```bash
   pytest tests/
   ```

See `docs/SPEC.md` for the language grammar and detailed module contracts before contributing.
