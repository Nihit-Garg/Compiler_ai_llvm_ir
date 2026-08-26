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
```

### `advisor/heuristics.py`
```python
from typing import List

def get_optimization_advice(ir: str) -> List[str]:
    """Analyzes the given IR and returns a list of suggested optimization passes."""
    pass
```

### `verifier/check.py`
```python
def verify_ir(ir: str) -> bool:
    """Verifies that the generated LLVM IR is well-formed and valid."""
    pass
```

### `backend/codegen.py`
```python
def generate_object_code(ir: str, target_triple: str) -> bytes:
    """Compiles the LLVM IR down to target machine object code."""
    pass
```
