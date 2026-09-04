"""
Recursive-descent parser for the MiniC compiler frontend.

Parses a list of Tokens (from the lexer) into an AST.
One method per grammar rule in docs/SPEC.md §1 (EBNF).
Interface defined in SPEC.md §4.
"""

from typing import List
from frontend.ast import (
    Token, ASTNode,
    Program, FunctionDef, Param, Block,
    VarDecl, Assignment, IfStmt, WhileStmt, ReturnStmt, ExprStmt,
    BinaryOp, NumberLiteral, Identifier, FunctionCall,
)
from frontend.errors import ParseError
from frontend.lexer import (
    INT_KW, IF_KW, ELSE_KW, WHILE_KW, RETURN_KW,
    IDENTIFIER, NUMBER,
    LPAREN, RPAREN, LBRACE, RBRACE, SEMICOLON, COMMA,
    ASSIGN, PLUS, MINUS, STAR, SLASH,
    EQ, NEQ, LT, LE, GT, GE,
    EOF,
)


# ---------------------------------------------------------------------------
# Internal Parser class
# ---------------------------------------------------------------------------

class _Parser:
    """Recursive-descent parser — one method per EBNF rule."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    # -- helpers --

    def _current(self) -> Token:
        """Return the current token without consuming it."""
        return self.tokens[self.pos]

    def _peek(self, offset: int = 1) -> Token:
        """Look ahead by `offset` tokens without consuming."""
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]  # EOF

    def _eat(self, expected_type: str) -> Token:
        """Consume and return the current token if it matches; else raise."""
        tok = self._current()
        if tok.type != expected_type:
            raise ParseError(
                f"Expected {expected_type} but got {tok.type} "
                f"('{tok.value}') at line {tok.line}, column {tok.column}"
            )
        self.pos += 1
        return tok

    def _match(self, *types: str) -> bool:
        """Check whether the current token is one of the given types."""
        return self._current().type in types

    # -- grammar rules --

    def parse_program(self) -> Program:
        """program ::= function_def*"""
        functions: List[FunctionDef] = []
        while not self._match(EOF):
            functions.append(self.parse_function_def())
        return Program(functions=functions)

    def parse_function_def(self) -> FunctionDef:
        """function_def ::= type_specifier IDENTIFIER '(' param_list? ')' block"""
        ret_type = self._eat(INT_KW).value           # type_specifier → 'int'
        name = self._eat(IDENTIFIER).value
        self._eat(LPAREN)
        params: List[Param] = []
        if not self._match(RPAREN):
            params = self.parse_param_list()
        self._eat(RPAREN)
        body = self.parse_block()
        return FunctionDef(return_type=ret_type, name=name, params=params, body=body)

    def parse_param_list(self) -> List[Param]:
        """param_list ::= type_specifier IDENTIFIER (',' type_specifier IDENTIFIER)*"""
        params: List[Param] = []
        p_type = self._eat(INT_KW).value
        p_name = self._eat(IDENTIFIER).value
        params.append(Param(type=p_type, name=p_name))
        while self._match(COMMA):
            self._eat(COMMA)
            p_type = self._eat(INT_KW).value
            p_name = self._eat(IDENTIFIER).value
            params.append(Param(type=p_type, name=p_name))
        return params

    def parse_block(self) -> Block:
        """block ::= '{' statement* '}'"""
        self._eat(LBRACE)
        stmts: List[ASTNode] = []
        while not self._match(RBRACE):
            stmts.append(self.parse_statement())
        self._eat(RBRACE)
        return Block(statements=stmts)

    def parse_statement(self) -> ASTNode:
        """
        statement ::= var_decl | assignment | if_stmt | while_stmt
                     | return_stmt | expr_stmt

        Disambiguation:
        - INT_KW          → var_decl
        - IF_KW           → if_stmt
        - WHILE_KW        → while_stmt
        - RETURN_KW       → return_stmt
        - IDENTIFIER + '=' (not '==') → assignment
        - otherwise        → expr_stmt
        """
        if self._match(INT_KW):
            return self.parse_var_decl()
        if self._match(IF_KW):
            return self.parse_if_stmt()
        if self._match(WHILE_KW):
            return self.parse_while_stmt()
        if self._match(RETURN_KW):
            return self.parse_return_stmt()

        # Disambiguate assignment vs expr_stmt:
        # assignment starts with IDENTIFIER '=' (where '=' is ASSIGN, not EQ)
        if self._match(IDENTIFIER) and self._peek().type == ASSIGN:
            return self.parse_assignment()

        return self.parse_expr_stmt()

    def parse_var_decl(self) -> VarDecl:
        """var_decl ::= type_specifier IDENTIFIER ('=' expr)? ';'"""
        v_type = self._eat(INT_KW).value
        v_name = self._eat(IDENTIFIER).value
        init = None
        if self._match(ASSIGN):
            self._eat(ASSIGN)
            init = self.parse_expr()
        self._eat(SEMICOLON)
        return VarDecl(type=v_type, name=v_name, init=init)

    def parse_assignment(self) -> Assignment:
        """assignment ::= IDENTIFIER '=' expr ';'"""
        name = self._eat(IDENTIFIER).value
        self._eat(ASSIGN)
        value = self.parse_expr()
        self._eat(SEMICOLON)
        return Assignment(name=name, value=value)

    def parse_if_stmt(self) -> IfStmt:
        """if_stmt ::= 'if' '(' expr ')' block ('else' block)?"""
        self._eat(IF_KW)
        self._eat(LPAREN)
        cond = self.parse_expr()
        self._eat(RPAREN)
        then_block = self.parse_block()
        else_block = None
        if self._match(ELSE_KW):
            self._eat(ELSE_KW)
            else_block = self.parse_block()
        return IfStmt(condition=cond, then_block=then_block, else_block=else_block)

    def parse_while_stmt(self) -> WhileStmt:
        """while_stmt ::= 'while' '(' expr ')' block"""
        self._eat(WHILE_KW)
        self._eat(LPAREN)
        cond = self.parse_expr()
        self._eat(RPAREN)
        body = self.parse_block()
        return WhileStmt(condition=cond, body=body)

    def parse_return_stmt(self) -> ReturnStmt:
        """return_stmt ::= 'return' expr ';'"""
        self._eat(RETURN_KW)
        value = self.parse_expr()
        self._eat(SEMICOLON)
        return ReturnStmt(value=value)

    def parse_expr_stmt(self) -> ExprStmt:
        """expr_stmt ::= expr? ';'"""
        expr = None
        if not self._match(SEMICOLON):
            expr = self.parse_expr()
        self._eat(SEMICOLON)
        return ExprStmt(expr=expr)

    # -- expression rules (precedence climbing) --

    def parse_expr(self) -> ASTNode:
        """expr ::= equality_expr"""
        return self.parse_equality_expr()

    def parse_equality_expr(self) -> ASTNode:
        """equality_expr ::= relational_expr (('==' | '!=') relational_expr)*"""
        left = self.parse_relational_expr()
        while self._match(EQ, NEQ):
            op = self._eat(self._current().type).value
            right = self.parse_relational_expr()
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def parse_relational_expr(self) -> ASTNode:
        """relational_expr ::= add_expr (('<' | '<=' | '>' | '>=') add_expr)*"""
        left = self.parse_add_expr()
        while self._match(LT, LE, GT, GE):
            op = self._eat(self._current().type).value
            right = self.parse_add_expr()
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def parse_add_expr(self) -> ASTNode:
        """add_expr ::= mult_expr (('+' | '-') mult_expr)*"""
        left = self.parse_mult_expr()
        while self._match(PLUS, MINUS):
            op = self._eat(self._current().type).value
            right = self.parse_mult_expr()
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def parse_mult_expr(self) -> ASTNode:
        """mult_expr ::= primary_expr (('*' | '/') primary_expr)*"""
        left = self.parse_primary_expr()
        while self._match(STAR, SLASH):
            op = self._eat(self._current().type).value
            right = self.parse_primary_expr()
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def parse_primary_expr(self) -> ASTNode:
        """
        primary_expr ::= IDENTIFIER | NUMBER | '(' expr ')' | function_call
        function_call ::= IDENTIFIER '(' arg_list? ')'

        Disambiguation: IDENTIFIER followed by '(' → function_call
        """
        # Number literal
        if self._match(NUMBER):
            tok = self._eat(NUMBER)
            return NumberLiteral(value=int(tok.value))

        # Parenthesized expression or function call or plain identifier
        if self._match(IDENTIFIER):
            # Lookahead: is this a function call?
            if self._peek().type == LPAREN:
                return self.parse_function_call()
            tok = self._eat(IDENTIFIER)
            return Identifier(name=tok.value)

        # Parenthesized expression
        if self._match(LPAREN):
            self._eat(LPAREN)
            expr = self.parse_expr()
            self._eat(RPAREN)
            return expr

        # Nothing matched — syntax error
        tok = self._current()
        raise ParseError(
            f"Unexpected token {tok.type} ('{tok.value}') "
            f"at line {tok.line}, column {tok.column}"
        )

    def parse_function_call(self) -> FunctionCall:
        """function_call ::= IDENTIFIER '(' arg_list? ')'"""
        name = self._eat(IDENTIFIER).value
        self._eat(LPAREN)
        args: List[ASTNode] = []
        if not self._match(RPAREN):
            args = self.parse_arg_list()
        self._eat(RPAREN)
        return FunctionCall(name=name, args=args)

    def parse_arg_list(self) -> List[ASTNode]:
        """arg_list ::= expr (',' expr)*"""
        args: List[ASTNode] = [self.parse_expr()]
        while self._match(COMMA):
            self._eat(COMMA)
            args.append(self.parse_expr())
        return args


# ---------------------------------------------------------------------------
# Public interface (SPEC.md §4)
# ---------------------------------------------------------------------------

def parse(tokens: List[Token]) -> ASTNode:
    """
    Parses a list of tokens into an Abstract Syntax Tree (AST).

    Args:
        tokens (List[Token]): The tokens produced by the lexer.

    Returns:
        ASTNode: The root node of the generated AST (a Program node).
    """
    parser = _Parser(tokens)
    return parser.parse_program()
