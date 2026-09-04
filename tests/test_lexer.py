"""
Unit tests for the MiniC lexer.

Each test verifies a specific token type or lexer behavior
as defined in docs/SPEC.md §1 grammar terminals.
"""

import os
import pytest
from frontend.lexer import (
    lex, LexerError,
    INT_KW, IF_KW, ELSE_KW, WHILE_KW, RETURN_KW,
    IDENTIFIER, NUMBER,
    LPAREN, RPAREN, LBRACE, RBRACE, SEMICOLON, COMMA,
    ASSIGN, PLUS, MINUS, STAR, SLASH,
    EQ, NEQ, LT, LE, GT, GE,
    EOF,
)

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def types(tokens):
    """Return a list of token types (excluding EOF)."""
    return [t.type for t in tokens if t.type != EOF]


def values(tokens):
    """Return a list of token values (excluding EOF)."""
    return [t.value for t in tokens if t.type != EOF]


# ---------------------------------------------------------------------------
# Basic token tests
# ---------------------------------------------------------------------------

class TestBasicTokens:
    def test_empty_input(self):
        tokens = lex("")
        assert len(tokens) == 1
        assert tokens[0].type == EOF

    def test_single_number(self):
        tokens = lex("42")
        assert types(tokens) == [NUMBER]
        assert values(tokens) == ["42"]

    def test_multi_digit_number(self):
        tokens = lex("12345")
        assert types(tokens) == [NUMBER]
        assert values(tokens) == ["12345"]

    def test_identifier(self):
        tokens = lex("foo")
        assert types(tokens) == [IDENTIFIER]
        assert values(tokens) == ["foo"]

    def test_identifier_with_underscore(self):
        tokens = lex("_bar_2")
        assert types(tokens) == [IDENTIFIER]
        assert values(tokens) == ["_bar_2"]


# ---------------------------------------------------------------------------
# Keyword tests
# ---------------------------------------------------------------------------

class TestKeywords:
    def test_int_keyword(self):
        tokens = lex("int")
        assert types(tokens) == [INT_KW]

    def test_if_keyword(self):
        tokens = lex("if")
        assert types(tokens) == [IF_KW]

    def test_else_keyword(self):
        tokens = lex("else")
        assert types(tokens) == [ELSE_KW]

    def test_while_keyword(self):
        tokens = lex("while")
        assert types(tokens) == [WHILE_KW]

    def test_return_keyword(self):
        tokens = lex("return")
        assert types(tokens) == [RETURN_KW]

    def test_keyword_prefix_is_identifier(self):
        """'integer' should be IDENTIFIER, not INT_KW."""
        tokens = lex("integer")
        assert types(tokens) == [IDENTIFIER]
        assert values(tokens) == ["integer"]


# ---------------------------------------------------------------------------
# Operator tests
# ---------------------------------------------------------------------------

class TestOperators:
    def test_all_single_char_operators(self):
        tokens = lex("+ - * / = < >")
        assert types(tokens) == [PLUS, MINUS, STAR, SLASH, ASSIGN, LT, GT]

    def test_all_two_char_operators(self):
        tokens = lex("== != <= >=")
        assert types(tokens) == [EQ, NEQ, LE, GE]

    def test_eq_vs_assign(self):
        """== must be distinguished from =."""
        tokens = lex("= ==")
        assert types(tokens) == [ASSIGN, EQ]

    def test_lt_vs_le(self):
        tokens = lex("< <=")
        assert types(tokens) == [LT, LE]

    def test_gt_vs_ge(self):
        tokens = lex("> >=")
        assert types(tokens) == [GT, GE]

    def test_neq(self):
        tokens = lex("!=")
        assert types(tokens) == [NEQ]


# ---------------------------------------------------------------------------
# Punctuation tests
# ---------------------------------------------------------------------------

class TestPunctuation:
    def test_all_punctuation(self):
        tokens = lex("( ) { } ; ,")
        assert types(tokens) == [LPAREN, RPAREN, LBRACE, RBRACE, SEMICOLON, COMMA]


# ---------------------------------------------------------------------------
# Whitespace & line tracking
# ---------------------------------------------------------------------------

class TestWhitespaceAndLines:
    def test_whitespace_skipped(self):
        tokens = lex("  42  ")
        assert types(tokens) == [NUMBER]

    def test_newline_increments_line(self):
        tokens = lex("42\n13")
        # First token on line 1, second on line 2
        assert tokens[0].line == 1
        assert tokens[1].line == 2

    def test_column_tracking(self):
        tokens = lex("int x")
        assert tokens[0].column == 1   # 'int' starts at col 1
        assert tokens[1].column == 5   # 'x' starts at col 5

    def test_tabs_handled(self):
        tokens = lex("\t42")
        assert types(tokens) == [NUMBER]


# ---------------------------------------------------------------------------
# Comment tests
# ---------------------------------------------------------------------------

class TestComments:
    def test_single_line_comment_skipped(self):
        tokens = lex("42 // this is ignored\n13")
        assert types(tokens) == [NUMBER, NUMBER]
        assert values(tokens) == ["42", "13"]

    def test_only_comment(self):
        tokens = lex("// nothing here")
        assert types(tokens) == []  # only EOF


# ---------------------------------------------------------------------------
# Error tests
# ---------------------------------------------------------------------------

class TestErrors:
    def test_unrecognized_at(self):
        with pytest.raises(LexerError):
            lex("@")

    def test_unrecognized_hash(self):
        with pytest.raises(LexerError):
            lex("#include")


# ---------------------------------------------------------------------------
# Sample file integration tests
# ---------------------------------------------------------------------------

class TestSampleFiles:
    def _lex_sample(self, filename):
        path = os.path.join(SAMPLES_DIR, filename)
        with open(path, "r") as f:
            source = f.read()
        return lex(source)

    def test_sample_01_return(self):
        """int main() { return 42; }"""
        tokens = self._lex_sample("01_return.c")
        assert types(tokens) == [
            INT_KW, IDENTIFIER, LPAREN, RPAREN, LBRACE,
            RETURN_KW, NUMBER, SEMICOLON,
            RBRACE,
        ]

    def test_sample_02_add(self):
        """int add(int a, int b) { int c = a + b; return c; }"""
        tokens = self._lex_sample("02_add.c")
        assert types(tokens) == [
            INT_KW, IDENTIFIER, LPAREN,
            INT_KW, IDENTIFIER, COMMA, INT_KW, IDENTIFIER,
            RPAREN, LBRACE,
            INT_KW, IDENTIFIER, ASSIGN, IDENTIFIER, PLUS, IDENTIFIER, SEMICOLON,
            RETURN_KW, IDENTIFIER, SEMICOLON,
            RBRACE,
        ]

    def test_sample_03_if_else(self):
        tokens = self._lex_sample("03_if_else.c")
        # Just verify it lexes without error and contains IF/ELSE keywords
        tt = types(tokens)
        assert IF_KW in tt
        assert ELSE_KW in tt

    def test_sample_04_while(self):
        tokens = self._lex_sample("04_while.c")
        tt = types(tokens)
        assert WHILE_KW in tt

    def test_sample_05_func_call(self):
        tokens = self._lex_sample("05_func_call.c")
        tt = types(tokens)
        # Two function definitions → 'int' should appear at least twice
        assert tt.count(INT_KW) >= 2
