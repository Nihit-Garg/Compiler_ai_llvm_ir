"""
Lexer (tokenizer) for the MiniC compiler frontend.

Tokenizes a MiniC source string into a list of Token objects.
Interface defined in docs/SPEC.md §4.
"""

from typing import List
from frontend.ast import Token
from frontend.errors import LexerError


# ---------------------------------------------------------------------------
# Token type constants
# ---------------------------------------------------------------------------

# Keywords
INT_KW    = "INT_KW"
IF_KW     = "IF_KW"
ELSE_KW   = "ELSE_KW"
WHILE_KW  = "WHILE_KW"
RETURN_KW = "RETURN_KW"

# Literals & identifiers
IDENTIFIER = "IDENTIFIER"
NUMBER     = "NUMBER"

# Punctuation
LPAREN    = "LPAREN"
RPAREN    = "RPAREN"
LBRACE    = "LBRACE"
RBRACE    = "RBRACE"
SEMICOLON = "SEMICOLON"
COMMA     = "COMMA"

# Operators
ASSIGN = "ASSIGN"   # =
PLUS   = "PLUS"     # +
MINUS  = "MINUS"    # -
STAR   = "STAR"     # *
SLASH  = "SLASH"    # /
EQ     = "EQ"       # ==
NEQ    = "NEQ"      # !=
LT     = "LT"       # <
LE     = "LE"       # <=
GT     = "GT"       # >
GE     = "GE"       # >=

# Special
EOF = "EOF"

# Keyword lookup table
_KEYWORDS = {
    "int":    INT_KW,
    "if":     IF_KW,
    "else":   ELSE_KW,
    "while":  WHILE_KW,
    "return": RETURN_KW,
}


# ---------------------------------------------------------------------------
# Public interface (SPEC.md §4)
# ---------------------------------------------------------------------------

def lex(source: str) -> List[Token]:
    """
    Tokenizes the given MiniC source string.

    Args:
        source (str): The raw MiniC source code.

    Returns:
        List[Token]: A list of tokens extracted from the source.

    Raises:
        LexerError: On unrecognized characters.
    """
    tokens: List[Token] = []
    pos = 0
    line = 1
    column = 1
    length = len(source)

    while pos < length:
        ch = source[pos]

        # --- Skip whitespace ---
        if ch in (' ', '\t', '\r'):
            pos += 1
            column += 1
            continue

        if ch == '\n':
            pos += 1
            line += 1
            column = 1
            continue

        # --- Skip single-line comments (// ...) ---
        if ch == '/' and pos + 1 < length and source[pos + 1] == '/':
            pos += 2
            while pos < length and source[pos] != '\n':
                pos += 1
            # Don't consume the newline — let the main loop handle it
            continue

        # --- Two-character operators (must check before single-char) ---
        if pos + 1 < length:
            two = source[pos:pos + 2]
            if two == '==':
                tokens.append(Token(EQ, '==', line, column))
                pos += 2
                column += 2
                continue
            if two == '!=':
                tokens.append(Token(NEQ, '!=', line, column))
                pos += 2
                column += 2
                continue
            if two == '<=':
                tokens.append(Token(LE, '<=', line, column))
                pos += 2
                column += 2
                continue
            if two == '>=':
                tokens.append(Token(GE, '>=', line, column))
                pos += 2
                column += 2
                continue

        # --- Single-character symbols ---
        single_map = {
            '(': LPAREN,
            ')': RPAREN,
            '{': LBRACE,
            '}': RBRACE,
            ';': SEMICOLON,
            ',': COMMA,
            '+': PLUS,
            '-': MINUS,
            '*': STAR,
            '/': SLASH,
            '=': ASSIGN,
            '<': LT,
            '>': GT,
        }
        if ch in single_map:
            tokens.append(Token(single_map[ch], ch, line, column))
            pos += 1
            column += 1
            continue

        # --- Number literal ---
        if ch.isdigit():
            start = pos
            start_col = column
            while pos < length and source[pos].isdigit():
                pos += 1
                column += 1
            tokens.append(Token(NUMBER, source[start:pos], line, start_col))
            continue

        # --- Identifier / keyword ---
        if ch.isalpha() or ch == '_':
            start = pos
            start_col = column
            while pos < length and (source[pos].isalnum() or source[pos] == '_'):
                pos += 1
                column += 1
            word = source[start:pos]
            tok_type = _KEYWORDS.get(word, IDENTIFIER)
            tokens.append(Token(tok_type, word, line, start_col))
            continue

        # --- Unrecognized character ---
        raise LexerError(
            f"Unexpected character '{ch}' at line {line}, column {column}"
        )

    # Append EOF
    tokens.append(Token(EOF, "", line, column))
    return tokens



