"""
AST node definitions for the MiniC compiler frontend.

Every node subclasses ASTNode. The Token dataclass is used by the lexer.
Node types map 1:1 to the EBNF grammar rules in docs/SPEC.md.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Token (used by the lexer — interface defined in SPEC.md §4)
# ---------------------------------------------------------------------------

@dataclass
class Token:
    type: str
    value: str
    line: int
    column: int


# ---------------------------------------------------------------------------
# AST base class
# ---------------------------------------------------------------------------

class ASTNode:
    """Base class for all AST nodes."""
    pass


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

@dataclass
class Program(ASTNode):
    """program ::= function_def*"""
    functions: List['FunctionDef'] = field(default_factory=list)


@dataclass
class Param(ASTNode):
    """One parameter in a param_list."""
    type: str       # always "int" in MiniC
    name: str


@dataclass
class FunctionDef(ASTNode):
    """function_def ::= type_specifier IDENTIFIER '(' param_list? ')' block"""
    return_type: str            # always "int" in MiniC
    name: str
    params: List[Param] = field(default_factory=list)
    body: 'Block' = None


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class Block(ASTNode):
    """block ::= '{' statement* '}'"""
    statements: List[ASTNode] = field(default_factory=list)


@dataclass
class VarDecl(ASTNode):
    """var_decl ::= type_specifier IDENTIFIER ('=' expr)? ';'"""
    type: str                   # always "int" in MiniC
    name: str
    init: Optional[ASTNode] = None


@dataclass
class Assignment(ASTNode):
    """assignment ::= IDENTIFIER '=' expr ';'"""
    name: str
    value: ASTNode = None


@dataclass
class IfStmt(ASTNode):
    """if_stmt ::= 'if' '(' expr ')' block ('else' block)?"""
    condition: ASTNode = None
    then_block: 'Block' = None
    else_block: Optional['Block'] = None


@dataclass
class WhileStmt(ASTNode):
    """while_stmt ::= 'while' '(' expr ')' block"""
    condition: ASTNode = None
    body: 'Block' = None


@dataclass
class ReturnStmt(ASTNode):
    """return_stmt ::= 'return' expr ';'"""
    value: ASTNode = None


@dataclass
class ExprStmt(ASTNode):
    """expr_stmt ::= expr? ';'"""
    expr: Optional[ASTNode] = None


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass
class BinaryOp(ASTNode):
    """Covers equality, relational, additive, and multiplicative operators."""
    op: str
    left: ASTNode = None
    right: ASTNode = None


@dataclass
class NumberLiteral(ASTNode):
    """NUMBER ::= [0-9]+"""
    value: int = 0


@dataclass
class Identifier(ASTNode):
    """IDENTIFIER ::= [a-zA-Z_][a-zA-Z0-9_]*"""
    name: str = ""


@dataclass
class FunctionCall(ASTNode):
    """function_call ::= IDENTIFIER '(' arg_list? ')'"""
    name: str = ""
    args: List[ASTNode] = field(default_factory=list)
