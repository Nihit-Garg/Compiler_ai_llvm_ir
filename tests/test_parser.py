"""
Unit tests for the MiniC recursive-descent parser.

Each test verifies a specific grammar rule from docs/SPEC.md §1.
Tests use the lexer to tokenize source, then parse into AST.
"""

import os
import pytest
from frontend.lexer import lex, EOF
from frontend.parser import parse, ParseError
from frontend.ast import (
    Program, FunctionDef, Param, Block,
    VarDecl, Assignment, IfStmt, WhileStmt, ReturnStmt, ExprStmt,
    BinaryOp, NumberLiteral, Identifier, FunctionCall,
    Token,
)

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


# ---------------------------------------------------------------------------
# Helper: lex + parse in one step
# ---------------------------------------------------------------------------

def parse_source(source: str):
    return parse(lex(source))


# ---------------------------------------------------------------------------
# program / function_def
# ---------------------------------------------------------------------------

class TestProgramAndFunction:
    def test_empty_program(self):
        """program ::= function_def* — zero functions is valid."""
        ast = parse([Token(EOF, "", 1, 1)])
        assert isinstance(ast, Program)
        assert ast.functions == []

    def test_simple_return(self):
        """int main() { return 42; }"""
        ast = parse_source("int main() { return 42; }")
        assert isinstance(ast, Program)
        assert len(ast.functions) == 1
        fn = ast.functions[0]
        assert isinstance(fn, FunctionDef)
        assert fn.return_type == "int"
        assert fn.name == "main"
        assert fn.params == []
        assert isinstance(fn.body, Block)

    def test_function_with_params(self):
        """int add(int a, int b) { return a; }"""
        ast = parse_source("int add(int a, int b) { return a; }")
        fn = ast.functions[0]
        assert len(fn.params) == 2
        assert fn.params[0] == Param(type="int", name="a")
        assert fn.params[1] == Param(type="int", name="b")

    def test_multiple_functions(self):
        src = """
        int double_it(int a) { return a; }
        int main() { return 0; }
        """
        ast = parse_source(src)
        assert len(ast.functions) == 2
        assert ast.functions[0].name == "double_it"
        assert ast.functions[1].name == "main"


# ---------------------------------------------------------------------------
# var_decl
# ---------------------------------------------------------------------------

class TestVarDecl:
    def test_var_decl_no_init(self):
        ast = parse_source("int main() { int x; return 0; }")
        stmt = ast.functions[0].body.statements[0]
        assert isinstance(stmt, VarDecl)
        assert stmt.type == "int"
        assert stmt.name == "x"
        assert stmt.init is None

    def test_var_decl_with_init(self):
        ast = parse_source("int main() { int x = 5; return x; }")
        stmt = ast.functions[0].body.statements[0]
        assert isinstance(stmt, VarDecl)
        assert stmt.name == "x"
        assert isinstance(stmt.init, NumberLiteral)
        assert stmt.init.value == 5


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------

class TestAssignment:
    def test_assignment(self):
        ast = parse_source("int main() { int x; x = 10; return x; }")
        stmt = ast.functions[0].body.statements[1]
        assert isinstance(stmt, Assignment)
        assert stmt.name == "x"
        assert isinstance(stmt.value, NumberLiteral)
        assert stmt.value.value == 10


# ---------------------------------------------------------------------------
# if_stmt
# ---------------------------------------------------------------------------

class TestIfStmt:
    def test_if_no_else(self):
        src = "int main() { int x; if (x > 0) { x = 1; } return x; }"
        ast = parse_source(src)
        stmt = ast.functions[0].body.statements[1]
        assert isinstance(stmt, IfStmt)
        assert isinstance(stmt.condition, BinaryOp)
        assert stmt.condition.op == ">"
        assert isinstance(stmt.then_block, Block)
        assert stmt.else_block is None

    def test_if_else(self):
        src = """
        int max(int x, int y) {
            if (x > y) { return x; } else { return y; }
        }
        """
        ast = parse_source(src)
        stmt = ast.functions[0].body.statements[0]
        assert isinstance(stmt, IfStmt)
        assert isinstance(stmt.then_block, Block)
        assert isinstance(stmt.else_block, Block)


# ---------------------------------------------------------------------------
# while_stmt
# ---------------------------------------------------------------------------

class TestWhileStmt:
    def test_while(self):
        src = "int main() { int i; i = 0; while (i < 10) { i = i + 1; } return i; }"
        ast = parse_source(src)
        stmts = ast.functions[0].body.statements
        w = stmts[2]
        assert isinstance(w, WhileStmt)
        assert isinstance(w.condition, BinaryOp)
        assert w.condition.op == "<"
        assert isinstance(w.body, Block)


# ---------------------------------------------------------------------------
# return_stmt
# ---------------------------------------------------------------------------

class TestReturnStmt:
    def test_return_number(self):
        ast = parse_source("int main() { return 42; }")
        stmt = ast.functions[0].body.statements[0]
        assert isinstance(stmt, ReturnStmt)
        assert isinstance(stmt.value, NumberLiteral)
        assert stmt.value.value == 42

    def test_return_identifier(self):
        ast = parse_source("int main() { int x; return x; }")
        stmt = ast.functions[0].body.statements[1]
        assert isinstance(stmt, ReturnStmt)
        assert isinstance(stmt.value, Identifier)


# ---------------------------------------------------------------------------
# expr_stmt
# ---------------------------------------------------------------------------

class TestExprStmt:
    def test_empty_expr_stmt(self):
        ast = parse_source("int main() { ; return 0; }")
        stmt = ast.functions[0].body.statements[0]
        assert isinstance(stmt, ExprStmt)
        assert stmt.expr is None

    def test_function_call_as_stmt(self):
        src = """
        int foo() { return 1; }
        int main() { foo(); return 0; }
        """
        ast = parse_source(src)
        stmt = ast.functions[1].body.statements[0]
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, FunctionCall)


# ---------------------------------------------------------------------------
# Expressions — operator precedence
# ---------------------------------------------------------------------------

class TestExpressions:
    def test_number_literal(self):
        ast = parse_source("int main() { return 7; }")
        ret = ast.functions[0].body.statements[0]
        assert isinstance(ret.value, NumberLiteral)
        assert ret.value.value == 7

    def test_addition(self):
        """a + b → BinaryOp('+', Identifier('a'), Identifier('b'))"""
        ast = parse_source("int f(int a, int b) { return a + b; }")
        ret = ast.functions[0].body.statements[0]
        expr = ret.value
        assert isinstance(expr, BinaryOp)
        assert expr.op == "+"

    def test_precedence_mult_over_add(self):
        """a + b * c → BinaryOp('+', a, BinaryOp('*', b, c))"""
        ast = parse_source("int f(int a, int b, int c) { return a + b * c; }")
        expr = ast.functions[0].body.statements[0].value
        assert isinstance(expr, BinaryOp)
        assert expr.op == "+"
        assert isinstance(expr.left, Identifier)
        assert isinstance(expr.right, BinaryOp)
        assert expr.right.op == "*"

    def test_parenthesized_expr(self):
        """(a + b) * c → BinaryOp('*', BinaryOp('+', a, b), c)"""
        ast = parse_source("int f(int a, int b, int c) { return (a + b) * c; }")
        expr = ast.functions[0].body.statements[0].value
        assert isinstance(expr, BinaryOp)
        assert expr.op == "*"
        assert isinstance(expr.left, BinaryOp)
        assert expr.left.op == "+"

    def test_equality_ops(self):
        ast = parse_source("int f(int a, int b) { return a == b; }")
        expr = ast.functions[0].body.statements[0].value
        assert isinstance(expr, BinaryOp)
        assert expr.op == "=="

    def test_neq_ops(self):
        ast = parse_source("int f(int a, int b) { return a != b; }")
        expr = ast.functions[0].body.statements[0].value
        assert expr.op == "!="

    def test_relational_ops(self):
        for op_src, op_str in [("<", "<"), ("<=", "<="), (">", ">"), (">=", ">=")]:
            ast = parse_source(f"int f(int a, int b) {{ return a {op_src} b; }}")
            expr = ast.functions[0].body.statements[0].value
            assert expr.op == op_str

    def test_function_call_expr(self):
        src = """
        int double_it(int a) { return a * 2; }
        int main() { int res = double_it(5); return res; }
        """
        ast = parse_source(src)
        decl = ast.functions[1].body.statements[0]
        assert isinstance(decl, VarDecl)
        assert isinstance(decl.init, FunctionCall)
        assert decl.init.name == "double_it"
        assert len(decl.init.args) == 1
        assert isinstance(decl.init.args[0], NumberLiteral)

    def test_function_call_multiple_args(self):
        src = """
        int add(int a, int b) { return a + b; }
        int main() { return add(1, 2); }
        """
        ast = parse_source(src)
        ret = ast.functions[1].body.statements[0]
        call = ret.value
        assert isinstance(call, FunctionCall)
        assert len(call.args) == 2


# ---------------------------------------------------------------------------
# Syntax error handling
# ---------------------------------------------------------------------------

class TestSyntaxErrors:
    def test_missing_semicolon(self):
        with pytest.raises(ParseError):
            parse_source("int main() { return 42 }")

    def test_missing_closing_brace(self):
        with pytest.raises(ParseError):
            parse_source("int main() { return 42;")

    def test_unexpected_token(self):
        with pytest.raises(ParseError):
            parse_source("int main() { return ; }")


# ---------------------------------------------------------------------------
# Sample file integration tests
# ---------------------------------------------------------------------------

class TestSampleFiles:
    def _parse_sample(self, filename):
        path = os.path.join(SAMPLES_DIR, filename)
        with open(path, "r") as f:
            source = f.read()
        return parse_source(source)

    def test_sample_01_return(self):
        ast = self._parse_sample("01_return.c")
        assert isinstance(ast, Program)
        assert len(ast.functions) == 1
        assert ast.functions[0].name == "main"

    def test_sample_02_add(self):
        ast = self._parse_sample("02_add.c")
        fn = ast.functions[0]
        assert fn.name == "add"
        assert len(fn.params) == 2

    def test_sample_03_if_else(self):
        ast = self._parse_sample("03_if_else.c")
        fn = ast.functions[0]
        assert fn.name == "max"
        # First statement should be an if-else
        stmt = fn.body.statements[0]
        assert isinstance(stmt, IfStmt)
        assert stmt.else_block is not None

    def test_sample_04_while(self):
        ast = self._parse_sample("04_while.c")
        fn = ast.functions[0]
        assert fn.name == "count_to"
        # Should contain a while statement
        has_while = any(isinstance(s, WhileStmt) for s in fn.body.statements)
        assert has_while

    def test_sample_05_func_call(self):
        ast = self._parse_sample("05_func_call.c")
        assert len(ast.functions) == 2
        assert ast.functions[0].name == "double_it"
        assert ast.functions[1].name == "main"
