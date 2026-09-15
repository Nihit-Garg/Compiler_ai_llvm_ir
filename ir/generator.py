"""
AST → LLVM IR generator for the MiniC compiler.

Translates the validated AST produced by the frontend into LLVM IR text.
The generated IR matches the format specified in docs/SPEC.md §3 and the
ground-truth samples in tests/samples/*.ll.

Interface defined in SPEC.md §4:
    generate_ir(ast: ASTNode) -> str

Design notes
------------
* Pure-Python text-based generation — no llvmlite dependency.
* One _FunctionBuilder per function manages SSA counters and block emission.
* All `alloca` instructions are hoisted to the entry block preamble (matching
  Clang's convention): first %retval (if needed), then parameter allocas, then
  all variable allocas declared anywhere in the function body.  This is
  accomplished with a two-pass scan: pass 1 collects every VarDecl name in
  declaration order; pass 2 emits IR.
* For if-else statements where both branches contain a `return`, a %retval
  alloca is hoisted to the entry block and a shared `return:` block is emitted
  at function end (matching SPEC §3 Example 3 and tests/samples/03_if_else.ll).
"""

from typing import List, Optional
from frontend.ast import (
    ASTNode, Program, FunctionDef, Param, Block,
    VarDecl, Assignment, IfStmt, WhileStmt, ReturnStmt, ExprStmt,
    BinaryOp, NumberLiteral, Identifier, FunctionCall,
)


# ---------------------------------------------------------------------------
# Operator mappings
# ---------------------------------------------------------------------------

_ARITH_OPS = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'sdiv'}

_CMP_OPS = {
    '<':  'slt',
    '<=': 'sle',
    '>':  'sgt',
    '>=': 'sge',
    '==': 'eq',
    '!=': 'ne',
}


# ---------------------------------------------------------------------------
# Static analysis helpers
# ---------------------------------------------------------------------------

def _block_always_returns(block: Block) -> bool:
    """Return True if the block unconditionally reaches a ReturnStmt."""
    if not block.statements:
        return False
    last = block.statements[-1]
    if isinstance(last, ReturnStmt):
        return True
    if isinstance(last, IfStmt) and last.else_block is not None:
        return (_block_always_returns(last.then_block) and
                _block_always_returns(last.else_block))
    return False


def _fn_needs_retval(fn: FunctionDef) -> bool:
    """True if any if-else in the function has both branches returning."""
    return _block_needs_retval(fn.body)


def _block_needs_retval(block: Block) -> bool:
    for stmt in block.statements:
        if isinstance(stmt, IfStmt):
            if (stmt.else_block is not None
                    and _block_always_returns(stmt.then_block)
                    and _block_always_returns(stmt.else_block)):
                return True
            if _block_needs_retval(stmt.then_block):
                return True
            if stmt.else_block and _block_needs_retval(stmt.else_block):
                return True
        elif isinstance(stmt, WhileStmt):
            if _block_needs_retval(stmt.body):
                return True
    return False


def _collect_var_decls(block: Block) -> List[str]:
    """
    Walk the block (and all nested blocks) in order, collecting VarDecl names
    in the order they appear.  Used to hoist allocas to the entry block.
    """
    names: List[str] = []
    for stmt in block.statements:
        if isinstance(stmt, VarDecl):
            names.append(stmt.name)
        elif isinstance(stmt, IfStmt):
            names.extend(_collect_var_decls(stmt.then_block))
            if stmt.else_block:
                names.extend(_collect_var_decls(stmt.else_block))
        elif isinstance(stmt, WhileStmt):
            names.extend(_collect_var_decls(stmt.body))
    return names


# ---------------------------------------------------------------------------
# Basic block accumulator
# ---------------------------------------------------------------------------

class _BasicBlock:
    def __init__(self, label: Optional[str]):
        self.label = label          # None → entry block (no printed label line)
        self.lines: List[str] = []

    def emit(self, line: str):
        self.lines.append(f"  {line}")

    def is_terminated(self) -> bool:
        for line in reversed(self.lines):
            s = line.strip()
            if s:
                return s.startswith("ret ") or s.startswith("br ")
        return False


# ---------------------------------------------------------------------------
# IR Builder — state for a single function
# ---------------------------------------------------------------------------

class _FunctionBuilder:
    def __init__(self):
        entry = _BasicBlock("entry")
        self._blocks: List[_BasicBlock] = [entry]
        self._current: _BasicBlock = entry

        self._tmp_counter: int = 0
        self._label_counters: dict = {}

        # name → alloca register (%x.addr for params, %x for vars)
        self.var_alloca: dict = {}

        self.retval: Optional[str] = None
        self.return_label: Optional[str] = None

    # ── SSA naming ──────────────────────────────────────────────────────────

    def fresh_tmp(self) -> str:
        """Anonymous SSA name: %0, %1, …"""
        name = f"%{self._tmp_counter}"
        self._tmp_counter += 1
        return name

    def fresh_label(self, kind: str) -> str:
        """Named label: first call → 'cmp', subsequent → 'cmp.1', 'cmp.2' …"""
        n = self._label_counters.get(kind, 0)
        self._label_counters[kind] = n + 1
        return kind if n == 0 else f"{kind}.{n}"

    # ── Block management ────────────────────────────────────────────────────

    def start_new_block(self, label: str):
        blk = _BasicBlock(label)
        self._blocks.append(blk)
        self._current = blk

    # ── Emission ────────────────────────────────────────────────────────────

    def emit(self, line: str):
        self._current.emit(line)

    def emit_to_entry(self, line: str):
        """Append a line directly to the entry block (for hoisted allocas)."""
        self._blocks[0].emit(line)

    # ── Rendering ───────────────────────────────────────────────────────────

    def render(self) -> List[str]:
        out: List[str] = []
        for i, blk in enumerate(self._blocks):
            if i == 0:
                # Entry block: print label without a preceding blank line
                out.append(f"{blk.label}:")
            else:
                # Non-entry blocks: blank line then label
                out.append("")
                out.append(f"{blk.label}:")
            out.extend(blk.lines)
        return out


# ---------------------------------------------------------------------------
# IR Generator visitor
# ---------------------------------------------------------------------------

class _IRGenerator:

    def visit_program(self, node: Program) -> str:
        parts: List[str] = []
        for i, fn in enumerate(node.functions):
            if i > 0:
                parts.append("")
            parts.append(self._visit_function_def(fn))
        return "\n".join(parts) + "\n"

    # ── Function ─────────────────────────────────────────────────────────────

    def _visit_function_def(self, node: FunctionDef) -> str:
        b = _FunctionBuilder()
        self._b = b

        needs_retval = _fn_needs_retval(node)
        if needs_retval:
            b.retval = "%retval"
            b.return_label = "return"

        # Build function signature
        param_strs = [f"i32 %{p.name}" for p in node.params]
        sig = f"define i32 @{node.name}({', '.join(param_strs)}) {{"

        # ── Hoist ALL allocas to entry block (Clang convention) ──────────────
        # Order: %retval (if needed), param allocas, then var allocas (in
        # declaration order across the whole function body).

        if needs_retval:
            b.emit_to_entry(f"{b.retval} = alloca i32")

        for p in node.params:
            alloca_name = f"%{p.name}.addr"
            b.var_alloca[p.name] = alloca_name
            b.emit_to_entry(f"{alloca_name} = alloca i32")

        var_names = _collect_var_decls(node.body)
        for vname in var_names:
            alloca_name = f"%{vname}"
            b.var_alloca[vname] = alloca_name
            b.emit_to_entry(f"{alloca_name} = alloca i32")

        # ── Store parameters into their allocas ──────────────────────────────
        for p in node.params:
            b.emit(f"store i32 %{p.name}, i32* {b.var_alloca[p.name]}")

        # ── Emit body ────────────────────────────────────────────────────────
        self._visit_block_stmts(node.body)

        # ── Shared return block (retval pattern) ─────────────────────────────
        if needs_retval:
            b.start_new_block(b.return_label)
            loaded = b.fresh_tmp()
            b.emit(f"{loaded} = load i32, i32* {b.retval}")
            b.emit(f"ret i32 {loaded}")

        lines = [sig] + b.render() + ["}"]
        self._b = None
        return "\n".join(lines)

    # ── Statements ────────────────────────────────────────────────────────────

    def _visit_block_stmts(self, block: Block):
        for stmt in block.statements:
            self._visit_statement(stmt)

    def _visit_statement(self, stmt: ASTNode):
        b = self._b

        if isinstance(stmt, VarDecl):
            # Alloca was already hoisted; just emit the init store (if any)
            if stmt.init is not None:
                alloca_name = b.var_alloca[stmt.name]
                val = self._visit_expr(stmt.init)
                b.emit(f"store i32 {val}, i32* {alloca_name}")

        elif isinstance(stmt, Assignment):
            alloca_name = b.var_alloca[stmt.name]
            val = self._visit_expr(stmt.value)
            b.emit(f"store i32 {val}, i32* {alloca_name}")

        elif isinstance(stmt, ReturnStmt):
            val = self._visit_expr(stmt.value)
            if b.retval is not None:
                b.emit(f"store i32 {val}, i32* {b.retval}")
                b.emit(f"br label %{b.return_label}")
            else:
                b.emit(f"ret i32 {val}")

        elif isinstance(stmt, IfStmt):
            self._visit_if(stmt)

        elif isinstance(stmt, WhileStmt):
            self._visit_while(stmt)

        elif isinstance(stmt, ExprStmt):
            if stmt.expr is not None:
                self._visit_expr(stmt.expr)

        else:
            raise NotImplementedError(f"Unhandled statement: {type(stmt).__name__}")

    def _visit_if(self, stmt: IfStmt):
        b = self._b

        then_label = b.fresh_label("if.then")
        else_label = b.fresh_label("if.else") if stmt.else_block else None
        end_label  = b.fresh_label("if.end")

        cmp_val = self._visit_expr(stmt.condition)
        false_target = else_label if stmt.else_block else end_label
        b.emit(f"br i1 {cmp_val}, label %{then_label}, label %{false_target}")

        # then branch
        b.start_new_block(then_label)
        self._visit_block_stmts(stmt.then_block)
        if not b._current.is_terminated():
            b.emit(f"br label %{end_label}")

        # else branch
        if stmt.else_block:
            b.start_new_block(else_label)
            self._visit_block_stmts(stmt.else_block)
            if not b._current.is_terminated():
                b.emit(f"br label %{end_label}")

        # merge block — skip if both branches always return (retval pattern)
        both_return = (
            stmt.else_block is not None
            and _block_always_returns(stmt.then_block)
            and _block_always_returns(stmt.else_block)
        )
        if not both_return:
            b.start_new_block(end_label)

    def _visit_while(self, stmt: WhileStmt):
        b = self._b

        cond_label = b.fresh_label("while.cond")
        body_label = b.fresh_label("while.body")
        end_label  = b.fresh_label("while.end")

        b.emit(f"br label %{cond_label}")

        b.start_new_block(cond_label)
        cmp_val = self._visit_expr(stmt.condition)
        b.emit(f"br i1 {cmp_val}, label %{body_label}, label %{end_label}")

        b.start_new_block(body_label)
        self._visit_block_stmts(stmt.body)
        if not b._current.is_terminated():
            b.emit(f"br label %{cond_label}")

        b.start_new_block(end_label)

    # ── Expressions (return SSA value string) ─────────────────────────────────

    def _visit_expr(self, expr: ASTNode) -> str:
        b = self._b

        if isinstance(expr, NumberLiteral):
            return str(expr.value)

        elif isinstance(expr, Identifier):
            alloca_name = b.var_alloca[expr.name]
            tmp = b.fresh_tmp()
            b.emit(f"{tmp} = load i32, i32* {alloca_name}")
            return tmp

        elif isinstance(expr, BinaryOp):
            op = expr.op
            if op in _CMP_OPS:
                left_val  = self._visit_expr(expr.left)
                right_val = self._visit_expr(expr.right)
                cmp_reg   = f"%{b.fresh_label('cmp')}"
                pred      = _CMP_OPS[op]
                b.emit(f"{cmp_reg} = icmp {pred} i32 {left_val}, {right_val}")
                return cmp_reg
            elif op in _ARITH_OPS:
                left_val  = self._visit_expr(expr.left)
                right_val = self._visit_expr(expr.right)
                instr     = _ARITH_OPS[op]
                result    = f"%{b.fresh_label(instr)}"
                b.emit(f"{result} = {instr} i32 {left_val}, {right_val}")
                return result
            else:
                raise NotImplementedError(f"Unknown operator: {op}")

        elif isinstance(expr, FunctionCall):
            arg_strs = [f"i32 {self._visit_expr(arg)}" for arg in expr.args]
            call_reg = f"%{b.fresh_label('call')}"
            b.emit(f"{call_reg} = call i32 @{expr.name}({', '.join(arg_strs)})")
            return call_reg

        else:
            raise NotImplementedError(f"Unhandled expression: {type(expr).__name__}")


# ---------------------------------------------------------------------------
# Public interface (SPEC.md §4)
# ---------------------------------------------------------------------------

def generate_ir(ast: ASTNode) -> str:
    """
    Translates the AST into an LLVM IR string.

    Args:
        ast (ASTNode): The root node of the validated AST (a Program node).

    Returns:
        str: The generated LLVM IR in textual format, matching the format
             specified in docs/SPEC.md §3.

    Raises:
        TypeError: If ast is not a Program node.
        NotImplementedError: If an unsupported AST node type is encountered.
    """
    if not isinstance(ast, Program):
        raise TypeError(f"generate_ir expects a Program node, got {type(ast).__name__}")
    gen = _IRGenerator()
    return gen.visit_program(ast)
