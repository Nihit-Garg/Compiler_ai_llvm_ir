"""
Unit tests for the MiniC semantic analyzer.

Tests symbol table resolution (declaring, using, scope tracking, shadowing),
function signatures, type checking (all expressions must be 'int'), and basic
control flow verification like return statements.
"""

import os
import pytest
from frontend.lexer import lex
from frontend.parser import parse
from frontend.semantic import analyze, SemanticError

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def analyze_source(source: str):
    """Lex -> Parse -> Analyze."""
    tokens = lex(source)
    ast = parse(tokens)
    return analyze(ast)


# ---------------------------------------------------------------------------
# Symbol Declaration and Scopes
# ---------------------------------------------------------------------------

class TestSymbolsAndScope:
    def test_undeclared_variable(self):
        src = "int main() { x = 5; return 0; }"
        with pytest.raises(SemanticError, match="Undeclared symbol 'x'"):
            analyze_source(src)

    def test_duplicate_declaration(self):
        src = "int main() { int x = 1; int x = 2; return 0; }"
        with pytest.raises(SemanticError, match="already declared in this scope"):
            analyze_source(src)

    def test_scope_shadowing_ok(self):
        src = """
        int main() {
            int x = 1;
            if (x > 0) {
                int x = 2; // Ok, different scope
                x = x + 1;
            }
            return x;
        }
        """
        # Should complete without error
        analyze_source(src)

    def test_scope_exit(self):
        src = """
        int main() {
            if (1 > 0) {
                int y = 5;
            }
            return y; // Error: y is not accessible here
        }
        """
        with pytest.raises(SemanticError, match="Undeclared symbol 'y'"):
            analyze_source(src)

    def test_param_scope(self):
        src = "int add(int a, int b) { return a + b; }"
        # Should not raise
        analyze_source(src)


# ---------------------------------------------------------------------------
# Function Calls
# ---------------------------------------------------------------------------

class TestFunctionCalls:
    def test_undeclared_function(self):
        src = "int main() { return foo(); }"
        with pytest.raises(SemanticError, match="Undeclared symbol 'foo'"):
            analyze_source(src)

    def test_wrong_arg_count_less(self):
        src = """
        int add(int a, int b) { return a + b; }
        int main() { return add(1); }
        """
        with pytest.raises(SemanticError, match="expects 2 arguments, but got 1"):
            analyze_source(src)

    def test_wrong_arg_count_more(self):
        src = """
        int foo() { return 1; }
        int main() { return foo(1, 2); }
        """
        with pytest.raises(SemanticError, match="expects 0 arguments, but got 2"):
            analyze_source(src)

    def test_forward_call_ok(self):
        """Pass 1 registers functions, so Pass 2 should allow calling forward."""
        src = """
        int main() { return foo(); }
        int foo() { return 42; }
        """
        # Should not raise
        analyze_source(src)


# ---------------------------------------------------------------------------
# Specific Errors
# ---------------------------------------------------------------------------

class TestSpecificErrors:
    def test_missing_return(self):
        src = "int main() { int x = 1; }"
        with pytest.raises(SemanticError, match="has no return statement"):
            analyze_source(src)

    def test_assign_to_function(self):
        src = "int foo() { return 1; } int main() { foo = 5; return 0; }"
        with pytest.raises(SemanticError, match="Cannot assign to 'foo', it is a function"):
            analyze_source(src)

    def test_function_as_value(self):
        src = "int foo() { return 1; } int main() { int x = foo; return 0; }"
        with pytest.raises(SemanticError, match="used as a value without being called"):
            analyze_source(src)


# ---------------------------------------------------------------------------
# Sample file integration tests
# ---------------------------------------------------------------------------

class TestSampleFiles:
    def _analyze_sample(self, filename):
        path = os.path.join(SAMPLES_DIR, filename)
        with open(path, "r") as f:
            source = f.read()
        return analyze_source(source)

    def test_sample_01_return(self):
        self._analyze_sample("01_return.c")

    def test_sample_02_add(self):
        self._analyze_sample("02_add.c")

    def test_sample_03_if_else(self):
        self._analyze_sample("03_if_else.c")

    def test_sample_04_while(self):
        self._analyze_sample("04_while.c")

    def test_sample_05_func_call(self):
        self._analyze_sample("05_func_call.c")
