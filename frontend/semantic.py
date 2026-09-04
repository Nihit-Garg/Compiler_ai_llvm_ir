"""
Semantic analyzer for the MiniC compiler frontend.

Performs symbol table management, scope resolution, and type checking on the AST.
Ensures variables and functions are declared before use, and that function
calls have the correct number of arguments.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from frontend.errors import SemanticError
from frontend.ast import (
    ASTNode, Program, FunctionDef, Param, Block,
    VarDecl, Assignment, IfStmt, WhileStmt, ReturnStmt, ExprStmt,
    BinaryOp, NumberLiteral, Identifier, FunctionCall
)


# ---------------------------------------------------------------------------
# Symbol Table & Scopes
# ---------------------------------------------------------------------------

@dataclass
class SymbolInfo:
    name: str
    type: str
    kind: str  # "variable", "parameter", "function"
    # For functions setting extra details:
    param_types: Optional[List[str]] = None
    return_type: Optional[str] = None


class SymbolTable:
    def __init__(self):
        # Stack of scopes. Each scope is a dict mapping name -> SymbolInfo
        self.scopes: List[Dict[str, SymbolInfo]] = []

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        if not self.scopes:
            raise SemanticError("Cannot exit scope: global scope reached.")
        self.scopes.pop()

    def declare(self, info: SymbolInfo):
        """Declare a symbol in the current scope."""
        if not self.scopes:
            raise SemanticError("No scope to declare symbol in.")
        current_scope = self.scopes[-1]
        if info.name in current_scope:
            raise SemanticError(f"Symbol '{info.name}' already declared in this scope.")
        current_scope[info.name] = info

    def lookup(self, name: str) -> Optional[SymbolInfo]:
        """Look up a symbol walking backwards from the current deepest scope."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def lookup_or_fail(self, name: str) -> SymbolInfo:
        info = self.lookup(name)
        if info is None:
            raise SemanticError(f"Undeclared symbol '{name}'.")
        return info


# ---------------------------------------------------------------------------
# Analyzer Visitor
# ---------------------------------------------------------------------------

class SemanticAnalyzer:
    def __init__(self):
        self.symtab = SymbolTable()
        self.symtab.enter_scope()  # Global scope
        self.current_function: Optional[FunctionDef] = None
        self.has_returned = False

    def analyze(self, ast: ASTNode) -> ASTNode:
        if isinstance(ast, Program):
            self.visit_program(ast)
        else:
            raise SemanticError("Semantic analysis must start at the Program root node.")
        return ast

    def visit_program(self, node: Program):
        # Pass 1: register all function signatures in the global scope
        for fn in node.functions:
            param_types = [p.type for p in fn.params]
            info = SymbolInfo(
                name=fn.name,
                type="function",
                kind="function",
                param_types=param_types,
                return_type=fn.return_type
            )
            self.symtab.declare(info)

        # Pass 2: thoroughly visit all function bodies
        for fn in node.functions:
            self.visit_function_def(fn)

    def visit_function_def(self, node: FunctionDef):
        self.current_function = node
        self.has_returned = False

        self.symtab.enter_scope()

        # Declare all parameters in the function's scope
        for p in node.params:
            info = SymbolInfo(name=p.name, type=p.type, kind="parameter")
            self.symtab.declare(info)

        # Visit body
        if node.body:
            self.visit_block(node.body, new_scope=False) # Function root already created the scope
        
        self.symtab.exit_scope()

        # Note: MiniC specification expects `int` return type for all functions.
        # But we do not strictly enforce `return` existence right now, though that 
        # is a good check to have for `int` non-void functions.
        if not self.has_returned and node.return_type != "void":
             raise SemanticError(f"Function '{node.name}' has no return statement.")
             
        self.current_function = None

    def visit_block(self, node: Block, new_scope: bool = True):
        if new_scope:
            self.symtab.enter_scope()
        
        for stmt in node.statements:
            self.visit_statement(stmt)
            
        if new_scope:
            self.symtab.exit_scope()

    def visit_statement(self, stmt: ASTNode):
        if isinstance(stmt, VarDecl):
            if stmt.init:
                self.visit_expr(stmt.init) # evaluate RHS first
            info = SymbolInfo(name=stmt.name, type=stmt.type, kind="variable")
            self.symtab.declare(info)
        elif isinstance(stmt, Assignment):
            info = self.symtab.lookup_or_fail(stmt.name)
            if info.kind not in ("variable", "parameter"):
                raise SemanticError(f"Cannot assign to '{stmt.name}', it is a {info.kind}.")
            self.visit_expr(stmt.value)
        elif isinstance(stmt, IfStmt):
            self.visit_expr(stmt.condition)
            self.visit_block(stmt.then_block)
            if stmt.else_block:
                self.visit_block(stmt.else_block)
        elif isinstance(stmt, WhileStmt):
            self.visit_expr(stmt.condition)
            self.visit_block(stmt.body)
        elif isinstance(stmt, ReturnStmt):
            self.visit_expr(stmt.value)
            self.has_returned = True
        elif isinstance(stmt, ExprStmt):
            if stmt.expr:
                self.visit_expr(stmt.expr)
        else:
            raise SemanticError(f"Unknown statement type {type(stmt).__name__}")

    def visit_expr(self, expr: ASTNode) -> str:
        """Returns the type of the expression."""
        if isinstance(expr, BinaryOp):
            left_type = self.visit_expr(expr.left)
            right_type = self.visit_expr(expr.right)
            if left_type != "int" or right_type != "int":
                raise SemanticError("Operands of binary operator must be of type 'int'.")
            return "int"
        elif isinstance(expr, NumberLiteral):
            return "int"
        elif isinstance(expr, Identifier):
            info = self.symtab.lookup_or_fail(expr.name)
            if info.kind == "function":
                 raise SemanticError(f"Function '{expr.name}' used as a value without being called.")
            return info.type
        elif isinstance(expr, FunctionCall):
            info = self.symtab.lookup_or_fail(expr.name)
            if info.kind != "function":
                raise SemanticError(f"'{expr.name}' is not a function and cannot be called.")
            
            if len(expr.args) != len(info.param_types):
                raise SemanticError(
                    f"Function '{expr.name}' expects {len(info.param_types)} arguments, "
                    f"but got {len(expr.args)}."
                )
            for arg in expr.args:
                arg_type = self.visit_expr(arg)
                if arg_type != "int":
                    raise SemanticError("Function arguments must be of type 'int'.")
            return info.return_type
        else:
            raise SemanticError(f"Unknown expression type {type(expr).__name__}")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def analyze(ast: ASTNode) -> ASTNode:
    """
    Performs semantic analysis on the given AST.
    Returns the validated and possibly annotated AST.
    Raises SemanticError if the program is semantically invalid.
    """
    analyzer = SemanticAnalyzer()
    return analyzer.analyze(ast)
