"""
Step-by-step display for the MiniC compiler (main.py --show-steps).

Prints what every stage of the pipeline produced, so a run can be followed
in the terminal: the source, the tokens, the AST, the symbols, the raw IR,
the IR after each optimization pass, the verifier's test runs and the final
IR.

Nothing here changes what the compiler does.  The per-pass IR is produced
with the same pass functions the pipeline uses, and the verifier table with
the verifier's own interpreter and test inputs, so what is printed is what
the pipeline computed.
"""

import sys
from typing import Dict, List, Optional, Tuple

from frontend.ast import (
    ASTNode, Program, FunctionDef, Block,
    VarDecl, Assignment, IfStmt, WhileStmt, ReturnStmt, ExprStmt,
    BinaryOp, NumberLiteral, Identifier, FunctionCall,
)
from verifier.pipeline import _PASS_FN
from verifier.check import (
    _parse_ir, _generate_inputs, _run_once, _MAX_STEPS, _MAX_CALL_DEPTH,
    count_instructions,
)


# Rows of the verifier table shown per function; the rest are summarised.
_MAX_VERIFIER_ROWS = 12


# ---------------------------------------------------------------------------
# Terminal colours (same fallback as main.py: plain text when not a tty)
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_DIM    = "\033[2m"


def _c(text: str, *codes: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _RESET


def _header(title: str):
    print()
    print(_c(f"  ┌─ {title} " + "─" * max(0, 52 - len(title)), _CYAN))


def _indent(text: str, prefix: str = "  │ "):
    for line in text.rstrip("\n").splitlines():
        print(prefix + line)


# ---------------------------------------------------------------------------
# Source and tokens
# ---------------------------------------------------------------------------

def show_source(source: str):
    _header("Input source code")
    lines = source.rstrip("\n").splitlines() or [""]
    width = len(str(len(lines)))
    for n, line in enumerate(lines, 1):
        print(f"  │ {_c(str(n).rjust(width), _DIM)}  {line}")


def show_tokens(tokens) -> None:
    _header(f"Step 1a: Lexer output ({len(tokens)} tokens)")
    print(f"  │ {_c('TYPE'.ljust(12), _BOLD)} {_c('VALUE'.ljust(12), _BOLD)} {_c('LINE:COL', _BOLD)}")
    for tok in tokens:
        value = tok.value if tok.value != "" else "-"
        print(f"  │ {tok.type.ljust(12)} {value.ljust(12)} {tok.line}:{tok.column}")


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

def _node_label(node: ASTNode) -> str:
    if isinstance(node, Program):
        return "Program"
    if isinstance(node, FunctionDef):
        params = ", ".join(f"{p.type} {p.name}" for p in node.params)
        return f"FunctionDef {node.return_type} {node.name}({params})"
    if isinstance(node, Block):
        return "Block"
    if isinstance(node, VarDecl):
        return f"VarDecl {node.type} {node.name}"
    if isinstance(node, Assignment):
        return f"Assignment {node.name} ="
    if isinstance(node, IfStmt):
        return "IfStmt"
    if isinstance(node, WhileStmt):
        return "WhileStmt"
    if isinstance(node, ReturnStmt):
        return "ReturnStmt"
    if isinstance(node, ExprStmt):
        return "ExprStmt"
    if isinstance(node, BinaryOp):
        return f"BinaryOp '{node.op}'"
    if isinstance(node, NumberLiteral):
        return f"Number {node.value}"
    if isinstance(node, Identifier):
        return f"Identifier {node.name}"
    if isinstance(node, FunctionCall):
        return f"Call {node.name}()"
    return type(node).__name__


def _node_children(node: ASTNode) -> List[Tuple[Optional[str], ASTNode]]:
    """(edge label, child) pairs; the edge label names the child's role."""
    if isinstance(node, Program):
        return [(None, f) for f in node.functions]
    if isinstance(node, FunctionDef):
        return [(None, node.body)] if node.body is not None else []
    if isinstance(node, Block):
        return [(None, s) for s in node.statements]
    if isinstance(node, VarDecl):
        return [("init", node.init)] if node.init is not None else []
    if isinstance(node, Assignment):
        return [("value", node.value)]
    if isinstance(node, IfStmt):
        kids = [("cond", node.condition), ("then", node.then_block)]
        if node.else_block is not None:
            kids.append(("else", node.else_block))
        return kids
    if isinstance(node, WhileStmt):
        return [("cond", node.condition), ("body", node.body)]
    if isinstance(node, ReturnStmt):
        return [(None, node.value)]
    if isinstance(node, ExprStmt):
        return [(None, node.expr)] if node.expr is not None else []
    if isinstance(node, BinaryOp):
        return [("left", node.left), ("right", node.right)]
    if isinstance(node, FunctionCall):
        return [(f"arg{i}", a) for i, a in enumerate(node.args)]
    return []


def _tree_lines(node: ASTNode, prefix: str = "", edge: Optional[str] = None,
                last: bool = True, root: bool = True) -> List[str]:
    label = _node_label(node)
    if edge:
        label = f"{_c(edge + ':', _DIM)} {label}"
    if root:
        lines = [label]
        child_prefix = ""
    else:
        lines = [prefix + ("└── " if last else "├── ") + label]
        child_prefix = prefix + ("    " if last else "│   ")
    kids = _node_children(node)
    for i, (kid_edge, kid) in enumerate(kids):
        lines += _tree_lines(kid, child_prefix, kid_edge, i == len(kids) - 1, False)
    return lines


def show_ast(ast: ASTNode):
    _header("Step 1b: Parser output (abstract syntax tree)")
    for line in _tree_lines(ast):
        print("  │ " + line)


# ---------------------------------------------------------------------------
# Symbols (after semantic analysis has accepted the program)
# ---------------------------------------------------------------------------

def _collect_locals(block: Block, depth: int, out: List[Tuple[str, int]]):
    for stmt in block.statements:
        if isinstance(stmt, VarDecl):
            out.append((stmt.name, depth))
        elif isinstance(stmt, IfStmt):
            _collect_locals(stmt.then_block, depth + 1, out)
            if stmt.else_block is not None:
                _collect_locals(stmt.else_block, depth + 1, out)
        elif isinstance(stmt, WhileStmt):
            _collect_locals(stmt.body, depth + 1, out)


def show_symbols(ast: Program):
    _header("Step 2: Semantic analysis (symbol table)")
    print("  │ " + _c("✓ no semantic errors", _GREEN))
    print("  │ " + _c("Functions:", _BOLD))
    for fn in ast.functions:
        params = ", ".join(p.name for p in fn.params) or "-"
        print(f"  │   {fn.name}: returns int, {len(fn.params)} parameter(s): {params}")
    for fn in ast.functions:
        local_vars: List[Tuple[str, int]] = []
        if fn.body is not None:
            _collect_locals(fn.body, 0, local_vars)
        print("  │ " + _c(f"Scope of {fn.name}:", _BOLD))
        for p in fn.params:
            print(f"  │   {p.name.ljust(10)} parameter")
        for name, depth in local_vars:
            where = "function body" if depth == 0 else f"inner block (depth {depth})"
            print(f"  │   {name.ljust(10)} local variable, {where}")
        if not fn.params and not local_vars:
            print("  │   (no variables)")


# ---------------------------------------------------------------------------
# IR
# ---------------------------------------------------------------------------

def show_ir(title: str, ir: str):
    _header(f"{title} ({count_instructions(ir)} instructions)")
    _indent(ir)


# ---------------------------------------------------------------------------
# Advisor, passes, verifier
# ---------------------------------------------------------------------------

def show_advisor(report: Dict):
    _header("Step 4a: LLM advisor")
    source = report.get("advisor_source")
    meaning = {
        "llm": "the LLM answered",
        "no_api_key": "GEMINI_API_KEY not set, default order used",
        "sdk_unavailable": "google-genai not installed, default order used",
        "api_error": "the API call failed, default order used",
        "unusable_response": "the LLM's answer had no valid pass list, default order used",
        "skipped": "advisor not asked (--no-advisor), default order used",
    }.get(source, "")
    print(f"  │ Source    : {source}  {_c('(' + meaning + ')', _DIM)}")
    if report.get("advisor_model"):
        print(f"  │ Model     : {report['advisor_model']}")
    passes = report.get("suggested_passes") or []
    print(f"  │ Suggested : {' → '.join(passes) if passes else '[] (no optimization)'}")


def show_pass_trace(title: str, raw_ir: str, passes: List[str]) -> str:
    """Apply the passes one at a time, printing the IR after each; return the result."""
    _header(title)
    ir = raw_ir
    print(f"  │ start: {count_instructions(ir)} instructions")
    for n, name in enumerate(passes, 1):
        fn = _PASS_FN.get(name)
        if fn is None:
            print(f"  │ {_c(f'pass {n}: {name} (unknown pass, skipped)', _YELLOW)}")
            continue
        before = count_instructions(ir)
        ir = fn(ir)
        after = count_instructions(ir)
        print("  │")
        print("  │ " + _c(f"after pass {n}: {name}  ({before} → {after} instructions)", _BOLD))
        _indent(ir, "  │     ")
    return ir


def _describe(kind: str, detail) -> str:
    if kind == "value":
        return str(detail)
    if kind == "trap":
        return f"trap ({detail})"
    return kind


def show_verifier(title: str, original_ir: str, optimized_ir: str):
    """Run the verifier's test inputs on both IRs and print the comparison."""
    _header(title)
    orig = _parse_ir(original_ir)
    opt = _parse_ir(optimized_ir)
    print("  │ Each function is run on the same inputs in the original and the")
    print("  │ optimized IR. The original's result is the expected answer.")
    for fn_name, fn in orig.functions.items():
        if fn_name not in opt.functions:
            print(f"  │ @{fn_name}: {_c('missing from the optimized IR', _RED)}")
            continue
        inputs = _generate_inputs(len(fn.params))
        params = ", ".join(fn.params) or "no parameters"
        print("  │")
        print("  │ " + _c(f"@{fn_name}({params}): {len(inputs)} test input(s)", _BOLD))
        print(f"  │   {'input'.ljust(18)} {'original'.ljust(14)} {'optimized'.ljust(14)} result")
        shown = matched = skipped = 0
        for args in inputs:
            ok, od = _run_once(orig, fn_name, args, _MAX_STEPS, _MAX_CALL_DEPTH)
            if ok == "inconclusive":
                skipped += 1
                verdict, pk, pd = _c("skipped (did not finish)", _YELLOW), "-", ""
            else:
                pk, pd = _run_once(opt, fn_name, args, _MAX_STEPS, _MAX_CALL_DEPTH)
                same = ok == pk and (ok != "value" or od == pd)
                matched += same
                verdict = _c("✓ same", _GREEN) if same else _c("✗ DIFFERENT", _RED, _BOLD)
            if shown < _MAX_VERIFIER_ROWS:
                arg_text = "(" + ", ".join(str(a) for a in args) + ")"
                opt_text = _describe(pk, pd) if pk != "-" else "-"
                print(f"  │   {arg_text.ljust(18)} {_describe(ok, od).ljust(14)} "
                      f"{opt_text.ljust(14)} {verdict}")
                shown += 1
        if len(inputs) > shown:
            print(f"  │   {_c(f'... {len(inputs) - shown} more input(s) not shown', _DIM)}")
        print(f"  │   {matched} matched, {skipped} skipped, "
              f"{len(inputs) - matched - skipped} different")


def show_decision(report: Dict):
    _header("Step 4d: Decision")
    decision = report.get("decision")
    colour = _GREEN if decision == "accepted" else _YELLOW
    print(f"  │ Suggestion : {_c(str(decision).upper(), colour, _BOLD)}")
    if report.get("rejection_reason"):
        print(f"  │ Reason     : {report['rejection_reason']}")
    if report.get("fallback_decision"):
        colour = _GREEN if report["fallback_decision"] == "accepted" else _YELLOW
        print(f"  │ Fallback   : {_c(report['fallback_decision'].upper(), colour, _BOLD)}"
              f"  (default order)")
        if report.get("fallback_rejection_reason"):
            print(f"  │ Reason     : {report['fallback_rejection_reason']}")
    applied = report.get("final_passes") or []
    print(f"  │ Applied    : {' → '.join(applied) if applied else 'none (IR left unoptimized)'}")
    print(f"  │ Size       : {report.get('instructions_before')} → "
          f"{report.get('instructions_after')} instructions")


def show_optimization(raw_ir: str, report: Dict):
    """Everything step 4 did: advisor answer, each pass, verifier runs, decision."""
    show_advisor(report)
    suggested = report.get("suggested_passes") or []
    if suggested:
        candidate = show_pass_trace("Step 4b: Applying the suggested passes one by one",
                                    raw_ir, suggested)
        show_verifier("Step 4c: Verifier (original vs. optimized)", raw_ir, candidate)
    fallback = report.get("fallback_passes")
    if fallback:
        candidate = show_pass_trace("Fallback: applying the default order one by one",
                                    raw_ir, fallback)
        show_verifier("Fallback: verifier (original vs. default order)", raw_ir, candidate)
    show_decision(report)


def show_final_check(check: Dict):
    _header("Step 4e: Final IR check")
    mark = lambda ok: _c("✓ passed", _GREEN) if ok else _c("✗ failed", _RED)
    print(f"  │ Structural check (verify_ir) : {mark(check.get('verify_ir'))}")
    llvm = check.get("llvm")
    if llvm == "unavailable":
        print(f"  │ LLVM verifier (llvmlite)     : {_c('not run (llvmlite not installed)', _YELLOW)}")
    else:
        print(f"  │ LLVM verifier (llvmlite)     : {mark(llvm == 'passed')}")
    if check.get("llvm_error"):
        print(f"  │ LLVM error: {check['llvm_error']}")


def show_object(obj_path, obj_bytes: bytes):
    _header("Step 5: Backend (object code)")
    print(f"  │ Wrote {len(obj_bytes)} bytes of native object code to {obj_path}")
