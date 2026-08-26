from dataclasses import dataclass
from typing import Any

@dataclass
class Token:
    type: str
    value: str
    line: int
    column: int

class ASTNode:
    """Base class for all AST nodes."""
    pass
