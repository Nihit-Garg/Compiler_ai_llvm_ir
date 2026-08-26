# Compiler AI LLVM IR

## Project Structure

- `/frontend`: Parsing and initial AST generation.
- `/ir`: Intermediate Representation definitions and translations.
- `/optimizer`: Optimization passes.
- `/advisor`: AI-assisted heuristics and advice for optimization.
- `/verifier`: Verification of IR and optimization correctness.
- `/backend`: Code generation for target architectures.
- `/tests`: Unit and integration tests.
- `/docs`: Documentation.

## Toolchain & Environment

To ensure compatibility, please use the following pinned toolchain versions:

- **Python**: `3.10` or higher
- **LLVM**: `14.0.x` (or `14.x` series)
- **llvmlite**: `0.40.0` (must match the LLVM 14 version)

### Setup Instructions

1. Ensure LLVM 14 is installed on your system.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install llvmlite pytest
   # OR if requirements.txt exists:
   # pip install -r requirements.txt
   ```

## Golden Rule for the Team (Preventing Hallucination)

**Context-first, always.**

Whenever you ask an LLM to generate code, you must **always** paste the following into the prompt:
1. `docs/SPEC.md` (for grammar and out-of-scope constraints).
2. The relevant skeleton/stub file (for the exact interface contract).
3. The relevant example test cases from `tests/samples/` (for grounded ground-truth).

**Never** ask an LLM to "build the parser" or "write an optimization pass" with no context — that's when hallucination happens.
