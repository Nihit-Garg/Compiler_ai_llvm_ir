"""
Custom exception hierarchy for the MiniC compiler frontend.
"""

class CompilerError(Exception):
    """Base class for all compiler errors."""
    pass

class LexerError(CompilerError):
    """Raised when the lexer encounters an unrecognized character."""
    pass

class ParseError(CompilerError):
    """Raised on syntax errors during parsing."""
    pass

class SemanticError(CompilerError):
    """Raised for any semantic/type-checking errors."""
    pass
