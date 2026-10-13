# Compiler AI LLVM IR Specification

This document serves as the single source-of-truth for the project. It defines the exact subset of C we are targeting, the expected LLVM IR format, the interface contracts for all modules, and explicitly states what is out of scope.

## 1. Language Grammar (MiniC Subset)

We are targeting a strict subset of C. The exact EBNF grammar is below:

```ebnf
program        ::= function_def*
function_def   ::= type_specifier IDENTIFIER "(" param_list? ")" block
type_specifier ::= "int"
param_list     ::= type_specifier IDENTIFIER ("," type_specifier IDENTIFIER)*

block          ::= "{" statement* "}"
statement      ::= var_decl | assignment | if_stmt | while_stmt | return_stmt | expr_stmt

var_decl       ::= type_specifier IDENTIFIER ("=" expr)? ";"
assignment     ::= IDENTIFIER "=" expr ";"
if_stmt        ::= "if" "(" expr ")" block ("else" block)?
while_stmt     ::= "while" "(" expr ")" block
return_stmt    ::= "return" expr ";"
expr_stmt      ::= expr? ";"

expr           ::= equality_expr
equality_expr  ::= relational_expr (("==" | "!=") relational_expr)*
relational_expr::= add_expr (("<" | "<=" | ">" | ">=") add_expr)*
add_expr       ::= mult_expr (("+" | "-") mult_expr)*
mult_expr      ::= primary_expr (("*" | "/") primary_expr)*
primary_expr   ::= IDENTIFIER | NUMBER | "(" expr ")" | function_call
function_call  ::= IDENTIFIER "(" arg_list? ")"
arg_list       ::= expr ("," expr)*

NUMBER         ::= [0-9]+
IDENTIFIER     ::= [a-zA-Z_][a-zA-Z0-9_]*
```

## 2. Out of Scope Features

To prevent scope creep and hallucination, the following features are **explicitly OUT OF SCOPE**:
- Pointers and memory addressing (`&`, `*`, `->`)
- Arrays and strings
- Structs, Unions, Enums
- Floating point numbers (`float`, `double`)
- Global variables
- `for` loops, `do-while` loops
- Preprocessor directives (`#include`, `#define`, macros)
- Logical operators (`&&`, `||`, `!`) - bitwise operators are also out of scope

## 3. Targeted LLVM IR Format & Examples

We are targeting LLVM IR. We will use standard `i32` for our `int` type.

### Example 1: Simple Return
**Source:**
```c
int main() {
    return 42;
}
```
**Expected LLVM IR:**
```llvm
define i32 @main() {
entry:
  ret i32 42
}
```

### Example 2: Variable Declaration and Addition
**Source:**
```c
int add(int a, int b) {
    int c = a + b;
    return c;
}
```
**Expected LLVM IR:**
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

### Example 3: If-Else Control Flow
**Source:**
```c
int max(int x, int y) {
    if (x > y) {
        return x;
    } else {
        return y;
    }
}
```
**Expected LLVM IR:**
```llvm
define i32 @max(i32 %x, i32 %y) {
entry:
  %retval = alloca i32
  %x.addr = alloca i32
  %y.addr = alloca i32
  store i32 %x, i32* %x.addr
  store i32 %y, i32* %y.addr
  %0 = load i32, i32* %x.addr
  %1 = load i32, i32* %y.addr
  %cmp = icmp sgt i32 %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %2 = load i32, i32* %x.addr
  store i32 %2, i32* %retval
  br label %return

if.else:
  %3 = load i32, i32* %y.addr
  store i32 %3, i32* %retval
  br label %return

return:
  %4 = load i32, i32* %retval
  ret i32 %4
}
```

### Generator rules

The examples above fix the output format. These rules cover what the examples do not show. IR that breaks any of them is rejected by LLVM.

**Types.** Every MiniC expression is an `i32`. An `icmp` yields an `i1`.

- A comparison used as a value (`int x = a < b;`, `return a == b;`, `(a < b) + 1`) is widened: `%conv = zext i1 %cmp to i32`.
- A non-comparison used as an `if` or `while` condition (`if (a)`, `while (n)`) is tested against zero: `%tobool = icmp ne i32 %0, 0`.
- A comparison used directly as a condition needs neither, as in Example 3.

**Terminators.** Every basic block ends in exactly one `ret` or `br`, and has none before its end.

- Statements after a `return` in the same block are unreachable and are not emitted.
- If control can run off the end of a function, the function returns 0 (`ret i32 0`, or a store of 0 to `%retval` followed by `br label %return`). C leaves this case undefined; MiniC defines it.
- A function with no `return` statement at all is still a semantic error.

**Names.** LLVM keeps the parameters, values and labels of a function in one namespace, so every name is unique within a function.

- Names are handed out in this order: `entry`, parameters, `%retval` and `return` (if used), parameter slots (`%x.addr`), variable slots in declaration order, then temporaries and labels as they are needed.
- A name that is already taken gets the next free suffix: `add`, `add.1`, `add.2`.
- So a MiniC variable named `add` keeps `%add` and the first addition becomes `%add.1`; a parameter named `entry` becomes `%entry.1`.

**Scopes.** Each declaration has its own stack slot.

- A declaration in an inner block that reuses a visible name gets a new slot (`%x.1`) and hides the outer variable until the block ends.
- An initializer is evaluated before its variable is declared: in an inner-block `int x = x + 1;` the right-hand `x` is the outer variable. This matches `frontend/semantic.py`.

## 4. Module Interface Contracts

All modules must strictly adhere to these interfaces. Teammates and their LLMs may fill in the bodies but CANNOT invent different interfaces.

### `frontend/lexer.py`
```python
from typing import List
from frontend.ast import Token

def lex(source: str) -> List[Token]:
    """Tokenizes the given MiniC source string."""
    pass
```

### `frontend/parser.py`
```python
from typing import List
from frontend.ast import Token, ASTNode

def parse(tokens: List[Token]) -> ASTNode:
    """Parses a list of tokens into an Abstract Syntax Tree (AST)."""
    pass
```

### `ir/generator.py`
```python
from frontend.ast import ASTNode

def generate_ir(ast: ASTNode) -> str:
    """Translates the AST into an LLVM IR string."""
    pass
```

### `optimizer/passes.py`
```python
def optimize(ir: str) -> str:
    """Applies optimization passes to the given LLVM IR string and returns the optimized IR."""
    pass

def constant_folding(ir: str) -> str:
    """Evaluates arithmetic instructions with literal operands."""
    pass

def dead_code_elimination(ir: str) -> str:
    """Removes unused instructions and unreachable basic blocks."""
    pass

def common_subexpression_elimination(ir: str) -> str:
    """Eliminates redundant identical instructions within basic blocks."""
    pass

def peephole(ir: str) -> str:
    """Applies small-window algebraic identities."""
    pass
```

### `advisor/heuristics.py`
```python
from typing import Any, Dict, List, Tuple

def get_optimization_advice(ir: str) -> List[str]:
    """
    Analyzes the given LLVM IR and returns an ordered list of optimization
    pass names to apply.  Falls back to the default fixed order on any error.
    Pass names match the function names in optimizer/passes.py exactly.
    """
    pass

def get_advice_with_source(ir: str) -> Tuple[List[str], Dict[str, Any]]:   # NEW
    """
    Same pass list as get_optimization_advice(), plus where it came from:
    {"source": str, "model": str | None}.
    """
    pass
```

Advisor rules:

- An explicit empty answer (`[]`) from the LLM is honoured: the result is `[]`, meaning "do not optimize". It is not treated as a failure.
- Any other unusable answer falls back to the default order: no JSON array, malformed JSON, or a non-empty array without a single valid pass name. Unknown names in an otherwise valid array are dropped.
- `source` is one of: `"llm"` (the LLM's own answer, including `[]`), `"no_api_key"`, `"sdk_unavailable"` (google-genai not installed), `"api_error"`, `"unusable_response"`. The pipeline adds `"skipped"` when the advisor is not consulted.
- `model` is the model name sent to the API, or `None` if no call was attempted.

### `verifier/check.py`
```python
from typing import Optional, Tuple

def verify_ir(ir: str) -> bool:
    """Structural well-formedness check. True if IR has valid shape."""
    pass

def verify_equivalence(original_ir: str, optimized_ir: str) -> bool:
    """
    Differential testing.  Interprets both IRs on concrete integer inputs.
    Returns True only if equivalence was established: identical outcomes on
    every tested input the original finishes on, with at least one such input
    per function.  False means the optimized IR must not be used.
    """
    pass

def equivalence_failure_reason(original_ir: str, optimized_ir: str) -> Optional[str]:
    """One-line reason why verify_equivalence() is False, or None if it is True."""
    pass

def count_instructions(ir: str) -> int:                       # NEW
    """Instructions in all functions; labels, comments and blank lines not counted."""
    pass

def llvm_check(ir: str) -> Tuple[str, Optional[str]]:         # NEW
    """
    Parse and verify the IR with LLVM via llvmlite.
    ("passed", None), ("failed", one-line LLVM error), or
    ("unavailable", None) if llvmlite cannot be imported.
    """
    pass
```

Verifier rules (the interpreter must follow LLVM's `i32` semantics):

- Arithmetic is 32-bit two's complement: `add`, `sub` and `mul` wrap; `sdiv` truncates toward zero.
- A run ends as a **value**, a **trap** (division by zero, or `INT_MIN / -1`) or **inconclusive** (step limit or call-depth limit reached).
- Value and trap are compared between the two IRs. An inconclusive run of the original is skipped and never counts as agreement.
- A function with no conclusive input is **not verified**.
- IR the interpreter cannot execute (unknown instruction, undefined register, block without terminator) is never equivalent.

The optimizer's constant folding must follow the same arithmetic rules, and must not fold a division that would trap.

### `verifier/pipeline.py` (NEW)
```python
from typing import Dict, Any, Tuple

def apply_optimizations(ir: str, use_advisor: bool = True) -> Tuple[str, Dict[str, Any]]:
    """
    Full advisor → verify → fallback pipeline.
    Returns (final_ir, report_dict).
    report_dict keys: suggested_passes, decision, rejection_reason,
                      fallback_passes, fallback_decision,
                      fallback_rejection_reason, final_passes,
                      advisor_source, advisor_model,
                      instructions_before, instructions_after.
    """
    pass
```

Pipeline rules:

- An empty suggestion (`[]`, the LLM's "no optimization") returns the input IR unchanged with `decision = "accepted"` and `final_passes = []`. Nothing was optimized, so there is nothing to verify and no fallback.
- The suggested pass order is verified against the original IR. If verified, it is used (`decision = "accepted"`).
- Otherwise (`decision = "rejected"`) the default pass order is applied **and verified the same way** (`fallback_decision`).
- If the default order is not verified either, the original IR is returned unchanged and `final_passes` is `[]`.
- `use_advisor=False` skips the LLM and uses the default order as the suggestion; verification still applies. `advisor_source` is then `"skipped"`.
- No path may emit optimized IR that the verifier has not passed.
- `advisor_source` / `advisor_model` come from `get_advice_with_source()`. They show whether `suggested_passes` is a real LLM answer or the default order used as a fallback.
- `instructions_before` / `instructions_after` are `count_instructions()` of the input IR and of the returned IR.

### `main.py` (command line)

```
python main.py <source.c> [-o FILE] [-r FILE] [--compile] [--no-advisor] [--quiet]
```

- The IR and the JSON report go to `build/<name>.ll` and `build/<name>_report.json`, with `build/` in the current working directory. They never go next to the source unless `-o` / `-r` say so. `--compile` writes `<ir path minus .ll>.o`.
- `--no-advisor` calls `apply_optimizations(ir, use_advisor=False)`. `--no-opt` is an alias kept for old scripts; despite its name, the default passes still run.
- Before reporting success, the final IR is checked with `verify_ir()`, and with `llvm_check()` when llvmlite is installed. The result goes in the report as `final_ir_check = {verify_ir, llvm, llvm_error, passed}`. A failed check still writes the IR so it can be inspected, skips `--compile`, and makes the exit status non-zero.
- Exit status: `0` on success. `1` for any error: unreadable source, a parse, semantic or IR-generation error, a failed final IR check, or a `--compile` that produced no object file.
- Besides the pipeline's keys, the report holds `source_file`, `ir_output`, `exit_code` and `elapsed_s`. It also holds `final_ir_check` (see above), `mode` (with `--no-advisor` only) and, with `--compile`, `object_output` and `compile_error`.

### `backend/codegen.py`
```python
def generate_object_code(ir: str, target_triple: str) -> bytes:
    """Compiles the LLVM IR down to target machine object code."""
    pass
```

## 5. Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | No | — | Gemini API key. Get free at https://aistudio.google.com/apikey. If unset, the advisor silently returns the default pass order. |
| `ADVISOR_MODEL` | No | `gemini-1.5-flash` | Gemini model name to use. |

The system is designed to work **without** a live API key — the advisor falls back to the fixed default pass order, the verifier still runs, and all tests pass. The API key is only required to actually exercise the LLM-guided path.
