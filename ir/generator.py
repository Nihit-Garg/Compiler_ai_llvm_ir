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
* Types: every MiniC expression is an `i32`.  A comparison yields an `i1`, so
  it is widened with `zext` when used as a value, and a plain integer used as
  a condition is tested with `icmp ne ..., 0`.  A comparison used directly as
  a condition needs neither.
* Every basic block ends in exactly one terminator.  Statements that follow a
  `return` in the same block are unreachable and are not emitted; a function
  whose control can run off its end returns 0.
* Names are unique per function.  LLVM keeps values and labels in one
  namespace, so a MiniC variable called `add`, `cmp`, `entry` or `retval`
  must not clash with a generated name; the later one gets a `.N` suffix.
* Each declaration has its own stack slot, so a variable that shadows another
  in an inner block does not overwrite it.
"""

from typing import List, Optional
from frontend.ast import (
    ASTNode, Program, FunctionDef, Block,
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

def _stmt_always_returns(stmt: ASTNode) -> bool:
    """True if control can never continue past this statement."""
    if isinstance(stmt, ReturnStmt):
        return True
    if isinstance(stmt, IfStmt) and stmt.else_block is not None:
        return (_block_always_returns(stmt.then_block) and
                _block_always_returns(stmt.else_block))
    return False


def _block_always_returns(block: Block) -> bool:
    """
    Return True if the block unconditionally reaches a ReturnStmt.

    Any statement that always returns is enough, not only the last one:
    whatever follows it is unreachable.
    """
    return any(_stmt_always_returns(stmt) for stmt in block.statements)


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


def _collect_var_decls(block: Block) -> List[VarDecl]:
    """
    Walk the block (and all nested blocks) in order, collecting VarDecl nodes
    in the order they appear.  Used to hoist allocas to the entry block.

    Nodes are returned rather than names because two declarations may share a
    name (an inner block shadowing an outer variable) and each needs its own
    stack slot.
    """
    decls: List[VarDecl] = []
    for stmt in block.statements:
        if isinstance(stmt, VarDecl):
            decls.append(stmt)
        elif isinstance(stmt, IfStmt):
            decls.extend(_collect_var_decls(stmt.then_block))
            if stmt.else_block:
                decls.extend(_collect_var_decls(stmt.else_block))
        elif isinstance(stmt, WhileStmt):
            decls.extend(_collect_var_decls(stmt.body))
    return decls


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

        # Every named value and label in the function, without the leading %.
        # LLVM keeps both in one namespace, so one set covers them all.
        self._used_names: set = {"entry"}
        self._name_counters: dict = {}

        # id(VarDecl node) → its alloca register; filled when allocas are hoisted
        self.decl_alloca: dict = {}

        # Lexical scopes, innermost last: MiniC name → alloca register
        # (%x.addr for params, %x for vars).  Mirrors frontend/semantic.py.
        self._scopes: List[dict] = [{}]

        self.retval: Optional[str] = None
        self.return_label: Optional[str] = None

    # ── SSA naming ──────────────────────────────────────────────────────────

    def fresh_tmp(self) -> str:
        """Anonymous SSA name: %0, %1, …"""
        name = f"%{self._tmp_counter}"
        self._tmp_counter += 1
        return name

    def fresh_label(self, kind: str) -> str:
        """
        A name not yet used in this function: first call → 'cmp', subsequent
        → 'cmp.1', 'cmp.2' …  Candidates already taken (for instance by a
        MiniC variable that happens to be called `cmp`) are skipped.
        Used for labels, named temporaries and stack slots alike.
        """
        n = self._name_counters.get(kind, 0)
        while True:
            name = kind if n == 0 else f"{kind}.{n}"
            n += 1
            if name not in self._used_names:
                break
        self._name_counters[kind] = n
        self._used_names.add(name)
        return name

    # ── Scopes ──────────────────────────────────────────────────────────────

    def push_scope(self):
        self._scopes.append({})

    def pop_scope(self):
        self._scopes.pop()

    def declare(self, name: str, alloca: str):
        """Bind a MiniC name to a stack slot in the innermost scope."""
        self._scopes[-1][name] = alloca

    def lookup(self, name: str) -> str:
        """Stack slot of the innermost visible declaration of `name`."""
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        raise KeyError(f"No stack slot for variable '{name}'")

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

        # Parameters are named first, so they keep their MiniC names.  The
        # one exception is a parameter called `entry`, which would clash with
        # the entry block's label and becomes `%entry.1`.
        param_regs = {p.name: f"%{b.fresh_label(p.name)}" for p in node.params}

        needs_retval = _fn_needs_retval(node)
        if needs_retval:
            b.retval = f"%{b.fresh_label('retval')}"
            b.return_label = b.fresh_label("return")

        # Build function signature
        param_strs = [f"i32 {param_regs[p.name]}" for p in node.params]
        sig = f"define i32 @{node.name}({', '.join(param_strs)}) {{"

        # ── Hoist ALL allocas to entry block (Clang convention) ──────────────
        # Order: %retval (if needed), param allocas, then var allocas (in
        # declaration order across the whole function body).

        if needs_retval:
            b.emit_to_entry(f"{b.retval} = alloca i32")

        for p in node.params:
            alloca_name = f"%{b.fresh_label(p.name + '.addr')}"
            b.declare(p.name, alloca_name)
            b.emit_to_entry(f"{alloca_name} = alloca i32")

        # One slot per declaration (not per name): a shadowing declaration in
        # an inner block gets `%x.1`, and so does a variable whose name is
        # already taken by a parameter or a generated name.
        for decl in _collect_var_decls(node.body):
            alloca_name = f"%{b.fresh_label(decl.name)}"
            b.decl_alloca[id(decl)] = alloca_name
            b.emit_to_entry(f"{alloca_name} = alloca i32")

        # ── Store parameters into their allocas ──────────────────────────────
        for p in node.params:
            b.emit(f"store i32 {param_regs[p.name]}, i32* {b.lookup(p.name)}")

        # ── Emit body ────────────────────────────────────────────────────────
        # The body shares the parameters' scope (as in frontend/semantic.py).
        self._visit_block_stmts(node.body, new_scope=False)

        # ── Control can run off the end of the function ──────────────────────
        # e.g. `if (a > 0) { return 1; }` with nothing after it, or the exit
        # block of a `while` that only ever leaves through a `return`.  The
        # block still needs a terminator; such a function returns 0.
        if not b._current.is_terminated():
            if needs_retval:
                b.emit(f"store i32 0, i32* {b.retval}")
                b.emit(f"br label %{b.return_label}")
            else:
                b.emit("ret i32 0")

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

    def _visit_block_stmts(self, block: Block, new_scope: bool = True):
        b = self._b
        if new_scope:
            b.push_scope()
        for stmt in block.statements:
            if b._current.is_terminated():
                # Everything after a `return` in this block is unreachable.
                # Emitting it would put instructions after the terminator.
                break
            self._visit_statement(stmt)
        if new_scope:
            b.pop_scope()

    def _visit_statement(self, stmt: ASTNode):
        b = self._b

        if isinstance(stmt, VarDecl):
            # Alloca was already hoisted; just emit the init store (if any).
            # The initializer is evaluated BEFORE the name is declared, so in
            # `int x = x + 1;` inside an inner block the right-hand `x` is the
            # outer variable.  frontend/semantic.py resolves it the same way.
            alloca_name = b.decl_alloca[id(stmt)]
            if stmt.init is not None:
                val = self._visit_expr(stmt.init)
                b.emit(f"store i32 {val}, i32* {alloca_name}")
            b.declare(stmt.name, alloca_name)

        elif isinstance(stmt, Assignment):
            alloca_name = b.lookup(stmt.name)
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

        cmp_val = self._visit_cond(stmt.condition)
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
        cmp_val = self._visit_cond(stmt.condition)
        b.emit(f"br i1 {cmp_val}, label %{body_label}, label %{end_label}")

        b.start_new_block(body_label)
        self._visit_block_stmts(stmt.body)
        if not b._current.is_terminated():
            b.emit(f"br label %{cond_label}")

        b.start_new_block(end_label)

    # ── Conditions (return an i1 value string) ────────────────────────────────

    def _emit_icmp(self, expr: BinaryOp) -> str:
        """Emit the `icmp` for a comparison and return its i1 register."""
        b = self._b
        left_val  = self._visit_expr(expr.left)
        right_val = self._visit_expr(expr.right)
        cmp_reg   = f"%{b.fresh_label('cmp')}"
        pred      = _CMP_OPS[expr.op]
        b.emit(f"{cmp_reg} = icmp {pred} i32 {left_val}, {right_val}")
        return cmp_reg

    def _visit_cond(self, expr: ASTNode) -> str:
        """
        Evaluate `expr` as the condition of an `if` or `while`: an i1.

        A comparison already is one.  Anything else is an i32 and, as in C,
        counts as true when it is not zero.
        """
        b = self._b
        if isinstance(expr, BinaryOp) and expr.op in _CMP_OPS:
            return self._emit_icmp(expr)
        val = self._visit_expr(expr)
        bool_reg = f"%{b.fresh_label('tobool')}"
        b.emit(f"{bool_reg} = icmp ne i32 {val}, 0")
        return bool_reg

    # ── Expressions (return an i32 value string) ──────────────────────────────

    def _visit_expr(self, expr: ASTNode) -> str:
        b = self._b

        if isinstance(expr, NumberLiteral):
            return str(expr.value)

        elif isinstance(expr, Identifier):
            alloca_name = b.lookup(expr.name)
            tmp = b.fresh_tmp()
            b.emit(f"{tmp} = load i32, i32* {alloca_name}")
            return tmp

        elif isinstance(expr, BinaryOp):
            op = expr.op
            if op in _CMP_OPS:
                # A comparison used as a value (`int x = a < b;`,
                # `return a == b;`) is 0 or 1 as an i32, as in C.
                cmp_reg  = self._emit_icmp(expr)
                conv_reg = f"%{b.fresh_label('conv')}"
                b.emit(f"{conv_reg} = zext i1 {cmp_reg} to i32")
                return conv_reg
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
