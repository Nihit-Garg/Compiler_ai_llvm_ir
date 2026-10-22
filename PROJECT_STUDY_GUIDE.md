# Verified LLM-Guided Compiler Optimization: Study Guide

Prepared 2026-10-21 for the project review.

---

## 1. The project in one paragraph

This project is a compiler for **MiniC**, a small subset of C. It turns MiniC into **LLVM IR**, then asks an LLM (Google Gemini) which optimization passes to run, and in what order, for *that particular program*. The LLM's answer is never trusted on its own. A **verifier** runs the original IR and the optimized IR on many integer inputs and compares the results. If any result differs, the suggestion is rejected and a fixed default order is tried, and that order is verified the same way. If the default order also fails, the unoptimized IR is kept.

**One-line pitch:** "An AI layer that can only make the compiler better, never incorrect."

**Headline result:** across 15 programs, the LLM's pass orders brought the total instruction count from 173 down to **130**. The fixed default order only reached 163. 130 is also the best that any of the 64 possible orderings reaches, so the LLM found the optimal order on every program.

**Why it matters:** GCC and Clang (`-O2`, `-O3`) run passes in one hard-coded order for every program. ML-guided optimizers such as AutoPhase and MLGO adapt the order to each program, but they give no guarantee that a suggestion keeps the program correct. This project does both at a small scale: it adapts per program and checks every result.

---

## 2. Background concepts you must be able to explain

### 2.1 LLVM IR

LLVM IR is the intermediate language that Clang compiles C into before it produces machine code. It reads like a typed assembly language with unlimited registers.

- **SSA (Static Single Assignment):** every `%name` is assigned exactly once. That is why the generator produces names like `%add`, `%add.1`, `%add.2`.
- **Basic block:** a straight run of instructions with one entry point (its label) and one exit (a terminator, either `ret` or `br`). Each block must end in exactly one terminator.
- **alloca / load / store:** local variables live in stack slots. `int c = a + b;` becomes `%c = alloca i32`, then a `store`, then a `load` when `c` is read. Clang produces the same shape at `-O0`.
- **icmp gives an i1:** a comparison produces a 1-bit boolean. When it is used as an `int`, it is widened with `zext i1 %cmp to i32`. When an `int` is used as a condition, it is tested with `icmp ne i32 %x, 0`.
- **Unnamed temporaries** (`%0, %1, ...`) must be numbered without gaps, or LLVM rejects the function. So any pass that deletes one has to renumber the rest (`_renumber_temporaries` in `optimizer/passes.py`).

Example from `docs/SPEC.md`:

```c
int add(int a, int b) { int c = a + b; return c; }
```

```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
  %a.addr = alloca i32
  %b.addr = alloca i32
  %c = alloca i32
  store i32 %a, i32* %a.addr
  store i32 %b, i32* %b.addr
  %0 = load i32, i32* %a.addr
  %1 = load i32, i32* %b.addr
  %add = add i32 %0, %1
  store i32 %add, i32* %c
  %2 = load i32, i32* %c
  ret i32 %2
}
```

### 2.2 The four optimization passes

A pass is one transformation over the IR. All four are written by the team in pure Python and work on IR **text**. They do not use LLVM's own optimizer.

| Pass | Abbrev. | What it does | Example |
|---|---|---|---|
| Constant folding | CF | Computes arithmetic whose two operands are literals, and puts the result into later uses | `add i32 3, 4` becomes `7` |
| Dead code elimination | DCE | Deletes instructions whose result is never used, and blocks that cannot be reached from `entry` | an unused `%t = mul ...` is removed |
| Common subexpression elimination | CSE | Inside one basic block, a repeated `op x, y` reuses the first result | the second `add %0, %1` is removed |
| Peephole | PH | Algebraic identities on a single instruction | `x+0 → x`, `x-0 → x`, `x*1 → x`, `x/1 → x`, `x*0 → 0`, `x*2 → x+x` |

A detail that matters for the results: **CF and PH do not delete the instruction they replace.** They leave it behind with a comment (`; folded -> 6`, `; peephole -> 0`), and it is now unused. Only DCE removes it. So DCE has to run *after* CF and PH to clean up.

### 2.3 The phase-ordering problem

The order in which passes run changes the result, because one pass creates work for another. This is a long-standing hard problem in compilers. In this project each pass runs at most once, so a pass order is an ordered selection of the 4 passes. There are 4 + 12 + 24 + 24 = **64** non-empty orderings.

The project built benchmarks where different programs need opposite orders:

- `b3`: `a * (3 - 2)` needs **CF first**. `3 - 2` folds to `1`, and then PH turns `a * 1` into `a`.
- `b8`: `(a * 0 + 2) * 3` needs **PH first**. `a * 0` becomes `0`, `0 + 2` becomes `2`, and then CF folds `2 * 3` to `6`.

No single fixed order is best for both programs. That is the argument for a per-program advisor.

### 2.4 Differential testing

Proving that two programs are equivalent is hard in general. Differential testing **runs both on the same inputs and compares the outputs**. If any input gives a different result, they are not equivalent. If every input agrees, you have strong evidence, but not a proof. The verifier uses this method.

---

## 3. Architecture

```
 MiniC source (.c)
      │
      ▼
 ┌──────────┐   ┌──────────┐   ┌───────────────┐   ┌──────────────┐
 │  Lexer   │──▶│  Parser  │──▶│   Semantic    │──▶│ IR generator │──▶ unoptimized IR
 │ tokens   │   │   AST    │   │ symbol table  │   │  LLVM IR text│
 └──────────┘   └──────────┘   └───────────────┘   └──────────────┘
                                                            │
      ┌─────────────────────────────────────────────────────┘
      ▼
 ┌────────────────┐  pass list   ┌──────────────┐ candidate IR ┌──────────────────┐
 │ LLM advisor    │─────────────▶│  Optimizer   │─────────────▶│ Verifier         │
 │ (Gemini)       │              │  4 passes    │              │ (differential    │
 └────────────────┘              └──────────────┘              │  testing)        │
                                                               └────────┬─────────┘
                       equivalent? ── yes ──▶ ACCEPT the LLM's IR       │
                            │                                           │
                            no                                          │
                            ▼                                           │
                apply DEFAULT order (CF → DCE → CSE → PH), verify it too
                            │
             equivalent? ── yes ──▶ use the default-order IR
                            │
                            no ──▶ keep the UNOPTIMIZED IR
                                          │
                                          ▼
              Final IR check: verify_ir() + LLVM's own verifier (llvmlite)
                                          │
                                          ▼
              Write build/<name>.ll + build/<name>_report.json
              Optional --compile: object file build/<name>.o via llvmlite
```

The safety guarantee is in `verifier/pipeline.py`: **no path emits optimized IR that the verifier has not passed.** Even the team's own default order gets no free pass.

---

## 4. Walkthrough of each stage

### 4.1 Frontend (`frontend/`)

- **Language (MiniC):** only `int`. It has functions, parameters, local variables, assignment, `if`/`else`, `while`, `return`, function calls (including recursion), `+ - * /`, and `== != < <= > >=`.
- **Out of scope:** pointers, arrays, strings, structs, floats, globals, `for`/`do-while`, the preprocessor, and the operators `&& || !` and bitwise operators.
- **Lexer** (`lexer.py`): a hand-written scanner. It turns source text into tokens (keywords, identifiers, numbers, operators) and raises `LexerError` on an unknown character.
- **Parser** (`parser.py`): a **recursive-descent** parser with one method per EBNF rule. Operator precedence comes from the grammar layers: equality, then relational, then add, then multiply, then primary. It builds the AST classes in `ast.py` and raises `ParseError`.
- **Semantic analysis** (`semantic.py`): a symbol table with nested scopes.
    - Every function signature is registered first, so a function can call one defined later in the file, and can call itself.
    - Names must be declared before use, and redeclaring a name in the same scope is an error. Parameters share a scope with the body's top-level declarations, and inner blocks may shadow outer names.
    - A call must pass exactly as many arguments as the function takes. A function name cannot be used as a value, and a variable cannot be called.
    - A function with no `return` at all is an error.
- **Errors** (`errors.py`): `CompilerError` is the base class, with `LexerError`, `ParseError` and `SemanticError` under it.

### 4.2 IR generation (`ir/generator.py`)

This stage writes LLVM IR as text. It does not use llvmlite.

- **Clang `-O0` style:** every variable gets an `alloca`, and all allocas are moved to the entry block. The generator makes two passes over each function: the first collects the declarations, the second emits the code.
- When a function has returns in more than one place (for example `if`/`else` that both return), it uses a `%retval` slot and one shared `return:` block, as Clang does.
- **Every block has exactly one terminator.** Code after a `return` is not emitted. If control can run off the end of a function, it returns 0 (C leaves this undefined; MiniC defines it).
- **Unique names:** if a MiniC variable is called `add`, it keeps `%add` and the first addition becomes `%add.1`.
- **Shadowing:** an inner `int x` gets its own slot `%x.1`.
- **Types:** `zext` turns a comparison into an `int`, and `icmp ne ..., 0` turns an `int` into a condition.
- The ground-truth outputs are in `tests/samples/*.ll`.

### 4.3 Optimizer (`optimizer/passes.py`)

The four passes from section 2.2, working on IR text with regular expressions. Correctness details worth mentioning in the review:

- **CF follows i32 rules:** results wrap at 32 bits (two's complement), and `sdiv` truncates toward zero. It **does not fold a division that would trap** (`x / 0`, or `INT_MIN / -1`).
- **DCE never deletes a `call` or a division that might trap**, even if the result is unused (`_must_keep`). Deleting one could remove an infinite loop or a crash, which would change what the program does.
- **DCE removes unreachable blocks** with a breadth-first search from `entry`.
- **CSE works within a single basic block.** It does not need dataflow analysis across blocks.
- After deleting instructions, the passes renumber `%0, %1, ...` so LLVM accepts the result.
- `optimize()` runs the default order: CF → DCE → CSE → PH.

### 4.4 LLM advisor (`advisor/heuristics.py`)

- **What it sends:** the IR, the 4 pass names **with a one-line description of each**, the goal ("fewest instructions"), and the rules ("return only a JSON array; each pass at most once; `[]` means no optimization").
- **Model:** `gemini-3.5-flash-lite` through the `google-genai` SDK. It can be changed with the `ADVISOR_MODEL` environment variable.
- **Parsing the answer:** it finds the JSON array in the reply, keeps only valid pass names, and drops duplicates. An explicit `[]` is honoured as "do not optimize".
- **Fallback:** on any failure, the advisor returns the default order and records why in `advisor_source`:

| `advisor_source` | Meaning |
|---|---|
| `llm` | the LLM's own answer was used (including `[]`) |
| `no_api_key` | `GEMINI_API_KEY` is not set |
| `sdk_unavailable` | `google-genai` is not installed |
| `api_error` | network, authentication or quota failure |
| `unusable_response` | no valid pass list in the reply |
| `skipped` | run with `--no-advisor` |

- **Key finding:** describing the passes in the prompt is what made the LLM good (see section 6).

### 4.5 Verifier (`verifier/check.py`)

This is the core of the safety argument.

- **A Python interpreter for the project's IR subset:** `alloca`, `load`, `store`, `add/sub/mul/sdiv`, `icmp`, `zext`, `br`, `ret` and `call`. Arithmetic follows LLVM's i32 rules.
- **Test inputs:** each parameter takes values from `[-7, -1, 0, 1, 2, 5, 42]`. These include negatives, zero, one and two, which are exactly the values the peephole rules depend on. Every combination is tried, up to 50 per function. When there are more combinations than that, it takes an evenly spaced sample so that every parameter still sees every value.
- **Each run ends in one of four ways:**
    - **value:** the function returned a number. The two IRs must return the same number.
    - **trap:** division by zero, or `INT_MIN / -1`. Both IRs must trap.
    - **inconclusive:** the run hit the 100,000-step limit or the call depth of 100. That input is skipped, and it never counts as agreement.
    - **malformed:** an unknown instruction or an undefined register. The IRs are never treated as equivalent.
- **A function with no conclusive input at all is NOT verified.** For example, a program that never terminates can never be optimized.
- **Clear failure messages,** for example `@diff(-7, -7): original returned 0, optimized returned -14`.
- **Other checks:**
    - `verify_ir()` checks the shape of the IR: there is at least one function, each has an entry block, and each contains a `ret`.
    - `llvm_check()` runs real LLVM's parser and verifier through llvmlite.
    - `count_instructions()` is the metric used in the results.

### 4.6 Pipeline (`verifier/pipeline.py`)

`apply_optimizations(ir, use_advisor=True)` returns `(final_ir, report)`. It follows the flow in section 3. If a pass or the verifier itself raises an exception, that counts as a rejection; it is never treated as a pass. The report records:

- what was suggested, the decision, and the reason for any rejection;
- the fallback passes, their decision, and the reason for any rejection;
- the passes actually applied;
- `advisor_source` and `advisor_model`;
- `instructions_before` and `instructions_after`.

### 4.7 Driver and backend

- **`main.py`** runs 5 stages and prints each one:
    1. lexing and parsing
    2. semantic analysis
    3. IR generation
    4. AI optimization and verification
    5. object code (only with `--compile`)
- Before it reports success, `main.py` runs the final IR check (`verify_ir` and `llvm_check`). It exits with 0 on success and 1 on any error.
- **`backend/codegen.py`** uses llvmlite to turn the IR into a native object file for the host machine (`.o`). It works with both old and new llvmlite, which handle `llvm.initialize()` differently.

### 4.8 Tests and CI

- There are **563 tests** (322 test functions, some run over several inputs) across 14 files in `tests/`. All 563 pass. They cover:
    - the lexer, parser and semantic analysis;
    - IR generation, checked against the `.ll` samples;
    - each optimizer pass, the advisor (with mocked LLM replies) and the verifier;
    - the pipeline, `main.py`, the backend, the evaluation, and that the docs match the code.
- **CI** (`.github/workflows/ci.yml`) runs the tests on two setups: Python 3.10 with llvmlite 0.40.0, and Python 3.12 with the latest llvmlite. No API key is needed, because the advisor falls back to the default order.

### 4.9 Project history (from git)

| Step | What was done |
|---|---|
| Base | Repo, README and `SPEC.md` (grammar, IR format, interfaces) |
| Person A | Frontend: lexer, parser, semantic analysis |
| Person B (navya-dev) | IR generator and optimizer passes, plus a DCE bug fix (a label lost its newline, which merged it with `ret`) |
| Person C | LLM advisor (Gemini), IR interpreter, differential verifier, fallback orchestrator |
| Integration | Backend (llvmlite), `main.py`, `.env` support, CI |
| Fix Phase 1 | Made the optimizer, verifier and fallback safe (i32 wrapping, trap handling, the default order is verified too) |
| Fix Phase 2 | Valid LLVM IR for every accepted program (terminators, unique names, shadowing, zext/icmp) |
| Fix Phases 3–4 | Command line, report, docs, toolchain |
| Phase 5 | Evaluation of the advisor and the safety net |
| Last | Benchmarks where the best order depends on the program (b8–b10), merged as PR #1 |

---

## 5. Code map

| File | Lines | Role |
|---|---:|---|
| `docs/SPEC.md` | 372 | Single source of truth: grammar, IR format, every module's interface |
| `frontend/lexer.py` | 186 | Tokenizer |
| `frontend/parser.py` | 304 | Recursive-descent parser |
| `frontend/ast.py` | 139 | AST node classes |
| `frontend/semantic.py` | 212 | Scopes, symbol table, semantic checks |
| `ir/generator.py` | 536 | AST to LLVM IR text |
| `optimizer/passes.py` | 674 | CF, DCE, CSE, peephole |
| `advisor/heuristics.py` | 289 | Gemini prompt, answer parsing, fallback |
| `verifier/check.py` | 620 | IR interpreter, differential testing, LLVM check |
| `verifier/pipeline.py` | 202 | Advisor → verify → fallback |
| `backend/codegen.py` | 115 | IR to object file (llvmlite) |
| `main.py` | 367 | Command-line driver and JSON report |
| `evaluation/run_eval.py` | 288 | Produces `evaluation/results.md` |

---

## 6. Evaluation and results

**Setup:** 15 programs (the 5 samples in `tests/samples/` and 10 benchmarks in `evaluation/benchmarks/`). For each program, the script counts instructions after four options:

1. no optimization;
2. the default order;
3. the LLM's order, after the verifier accepted it;
4. the best of all 64 orderings, found by brute force, with each candidate verified.

| Program | Unoptimized | Default | LLM | Best | LLM's order |
|---|---:|---:|---:|---:|---|
| 01_return | 1 | 1 | 1 | 1 | (none) |
| 02_add | 11 | 11 | 11 | 11 | (none) |
| 03_if_else | 17 | 17 | 17 | 17 | (none) |
| 04_while | 15 | 15 | 15 | 15 | (none) |
| 05_func_call | 10 | 10 | 10 | 10 | PH → DCE |
| b1_constant_chain | 5 | 1 | 1 | 1 | CF → DCE |
| b2_identities | 6 | 6 | **4** | 4 | PH → DCE |
| b3_fold_then_simplify | 10 | 8 | **4** | 4 | CF → PH → DCE |
| b4_mixed | 12 | 11 | **9** | 9 | PH → CF → CSE → DCE |
| b5_two_params | 13 | 13 | **12** | 12 | PH → CSE → DCE |
| b6_loop | 23 | 23 | **21** | 21 | PH → DCE |
| b7_calls | 16 | 13 | **10** | 10 | CF → PH → DCE |
| b8_simplify_then_fold | 7 | 7 | **3** | 3 | PH → CF → DCE |
| b9_simplify_then_fold_two | 12 | 12 | **5** | 5 | PH → CF → DCE |
| b10_simplify_then_fold_div | 15 | 15 | **7** | 7 | PH → CF → DCE |
| **Total** | **173** | **163** | **130** | **130** | |

**What to say:**

- The default order removed 10 instructions. The LLM's orders removed **43**, the same as the brute-force best.
- The LLM beat the default on 9 programs, tied on 6, and was **never worse**. It found the best order on all 15.
- **Why the default order is weak:** it runs DCE *before* PH, so the instructions PH leaves behind are never removed.
- **The LLM adapted to each program:** it suggested CF → PH → DCE for b3 and PH → CF → DCE for b8–b10, which are opposite orders.
- **Prompt engineering mattered.** In the first run the LLM saw only the pass names. On the 12 programs that run covered, it beat the default on 3 and reached the best on 8 of the 9 it answered. After a one-line description of each pass and the goal were added to the prompt, it beat the default on 6 of those 12 and reached the best on all 12. The names-only results are in `evaluation/results_names_only.md`.

**Worked example (b8: `return (a * 0 + 2) * 3;`), generated now from the code:**

Unoptimized (7 instructions):
```llvm
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  %mul = mul i32 %0, 0
  %add = add i32 %mul, 2
  %mul.1 = mul i32 %add, 3
  ret i32 %mul.1
```
Default order CF → DCE → CSE → PH (still 7): CF found nothing to fold, DCE ran too early, and PH then left dead lines and even turned `3 * 2` into `add 3, 3`:
```llvm
  %0 = load i32, i32* %a.addr
  %mul = mul i32 %0, 0  ; peephole -> 0
  %add = add i32 0, 2  ; peephole -> 2
  %mul.1 = add i32 3, 3
  ret i32 %mul.1
```
LLM order PH → CF → DCE (3):
```llvm
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  ret i32 6
```

**Safety net tests** (`run_eval.py` runs the real pipeline with a simulated LLM):

| Case | What happened |
|---|---|
| The LLM suggests a deliberately broken pass (turns `sub` into `add`) | Rejected with `@diff(-7, -7): original returned 0, optimized returned -14`. The default order is verified and used instead. |
| A program that never terminates | Both the suggestion and the fallback are rejected as inconclusive, and the IR is left unchanged. |
| The LLM replies with prose instead of JSON | Recorded as `unusable_response`. The default order is used after it is verified. |
| The LLM replies `[]` | No passes are applied, and the IR is unchanged. |

In every case, the IR that comes out is equivalent to the IR that went in.

---

## 7. How to run a demo

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest tests/                                   # full test suite, no API key needed

python main.py evaluation/benchmarks/b8_simplify_then_fold.c        # with the LLM, if .env has GEMINI_API_KEY
python main.py evaluation/benchmarks/b8_simplify_then_fold.c --no-advisor   # default order, still verified
python main.py tests/samples/02_add.c --compile                     # also writes build/02_add.o

python evaluation/run_eval.py          # rebuild the results tables from saved LLM answers (no API key)
python evaluation/run_eval.py --live   # ask Gemini again (slow: 30–100 s per program)
```

**Show every step (for the teacher):**

```bash
python main.py evaluation/benchmarks/b8_simplify_then_fold.c --show-steps
python main.py --interactive --show-steps     # type a program, then press Ctrl-D
```

This prints the source, tokens, AST, symbol table, unoptimized IR, the IR after each pass with its instruction count, the verifier's table (each input with the original result, the optimized result and ✓ or ✗), the decision, the final check and the final IR. Typed programs are written to `build/input.ll`.

Outputs go to `build/<name>.ll` and `build/<name>_report.json`. Open the JSON report during the demo: it shows `suggested_passes`, `decision`, `advisor_source` and `instructions_before` / `instructions_after`.

**Before the demo:**

- Run the `pip install` step above first.
- Never use `-o` to write output into `tests/samples/`. The `.ll` files there are the expected results the tests compare against, and overwriting one makes 2 tests fail.

---

## 8. Likely questions and good answers

**Q: Why not just use LLVM's `opt` with `-O2`?**
`-O2` uses one fixed order for every program. The project studies whether a per-program order chosen by an LLM can beat a fixed order *safely*. Writing the passes ourselves keeps them small enough to explain, and lets the LLM's choice clearly decide the result.

**Q: Is the verifier a proof of correctness?**
No. It is differential testing on sample inputs, so it is strong evidence, not a proof. Two safeguards keep it conservative:
- Inputs that never finish count for nothing, and a function with no conclusive input is rejected.
- Anything the interpreter does not understand is rejected.

The final IR is also checked by LLVM's own verifier. A future step would be translation validation, for example with an SMT solver such as Z3, or Alive2.

**Q: What if a bug only shows on an input you did not test?**
That is possible. The test values are chosen to cover the cases that matter for these passes: 0, 1 and 2 (the peephole identities), negatives (sign and division), and a larger value. The sample of combinations makes sure every parameter sees every value. More inputs (random or boundary values such as INT_MAX) would make it stronger.

**Q: Why does the LLM help at all? Couldn't you just brute-force the order?**
For 4 passes, yes: 64 orders is cheap, and the evaluation does exactly that to find the best possible result. Real compilers have more than 100 passes that can repeat, so brute force is impossible there. The LLM is a heuristic that scales. In this project it matched the brute-force optimum on all 15 programs.

**Q: Why does the LLM say "no optimization" for samples 01–04?**
Because there is nothing to gain. The IR keeps variables in `alloca` slots and reads them with `load`, so constant folding cannot see through variables. Without a `mem2reg` pass (promoting memory to registers), the samples have no foldable constants or identities. The LLM's `[]` was correct, since the best possible result equals the unoptimized count.

**Q: What are the limitations?**
- The verifier is testing, not proof.
- The only measure is the instruction count; runtime is not timed.
- The benchmarks are small, and some were written to show the ordering effect.
- There is no `mem2reg`, and the passes are text and regex based, written for this IR format.
- Each pass runs at most once.
- Each LLM call took 30–100 seconds, and a few calls failed with API errors. Those programs fell back to the default order and were asked again (`--retry-errors`).
- The LLM's answer can vary from run to run, so the evaluation saves the answers it got for reproducibility.

**Q: How do you make sure an unusable LLM answer cannot break things?**
- The parser only accepts known pass names in a JSON array, with no duplicates.
- Anything else falls back to the default order.
- Every path goes through the verifier before any optimized IR is emitted.
- An exception inside a pass or the verifier counts as a rejection.

**Q: What would you add next?**
- `mem2reg`, and more passes (loop-invariant code motion, copy propagation).
- Repeated passes, or iterating until nothing changes.
- Larger real benchmarks, and measuring runtime.
- SMT-based verification.
- Comparing against LLVM's own `-O2` on the same IR.
- Comparing the LLM against cheaper search methods (greedy, genetic).

**Q: How did you keep the LLM-written code consistent across the team?**
`docs/SPEC.md` fixed the grammar, the IR format and every function signature before any code was written. The README rule was: give the LLM the spec section, the interface stub and a sample before asking it for code. Tests (`tests/test_docs.py`) also check that the documentation matches the code.
