"""
End-to-end integration tests for the MiniC frontend pipeline.
Pipeline: lex -> parse -> analyze.

Iterates over all sample programs in the tests/samples/ directory
and verifies that they pass through the entire frontend without error.
"""

import os
import glob
import pytest
from frontend.lexer import lex
from frontend.parser import parse
from frontend.semantic import analyze

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

def get_sample_files():
    """Finds all .c files in the samples directory."""
    files = glob.glob(os.path.join(SAMPLES_DIR, "*.c"))
    # Sort for deterministic test output
    return sorted(files)

@pytest.mark.parametrize("filepath", get_sample_files())
def test_frontend_pipeline(filepath):
    """
    Test that a valid sample program successfully completes
    the source -> Lexer -> Parser -> Semantic Analyzer pipeline.
    """
    with open(filepath, "r") as f:
        source = f.read()

    try:
        # Phase 1 & 2 (Lexing implicitly creates Token AST nodes)
        tokens = lex(source)
        
        # Phase 3
        ast = parse(tokens)
        
        # Phase 4
        annotated_ast = analyze(ast)
        
        # If we reach here, it means no LexerError, ParseError,
        # or SemanticError was raised, which is the expected result
        # for ground-truth sample programs.
        assert annotated_ast is not None

    except Exception as e:
        pytest.fail(f"Pipeline failed on {os.path.basename(filepath)} with error: {e}")
