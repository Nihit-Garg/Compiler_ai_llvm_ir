"""
Fallback Orchestrator for the MiniC compiler.

This module is the integration point between the AI advisor, the optimizer
passes, and the safety verifier.  It implements the core safety guarantee:

    "The AI layer can only make the compiler better — never incorrect."

Public interface:

    apply_optimizations(ir: str) -> tuple[str, dict]
        1. Ask the LLM advisor for a suggested pass sequence.
        2. Apply the suggested passes to produce candidate IR.
        3. Run the differential verifier (original vs candidate).
        4a. If verified → accept, return candidate IR.
        4b. If rejected → apply the fixed default pass order, return that.
        Return the final IR together with a report dict for logging/evaluation.
"""

from typing import Dict, Any, List, Tuple

from advisor.heuristics import get_optimization_advice, _default_passes, _VALID_PASSES
from optimizer.passes import (
    constant_folding,
    dead_code_elimination,
    common_subexpression_elimination,
    peephole,
)
from verifier.check import verify_equivalence


# ---------------------------------------------------------------------------
# Pass dispatcher
# ---------------------------------------------------------------------------

_PASS_FN = {
    "constant_folding":              constant_folding,
    "dead_code_elimination":         dead_code_elimination,
    "common_subexpression_elimination": common_subexpression_elimination,
    "peephole":                      peephole,
}


def _apply_pass_sequence(ir: str, passes: List[str]) -> str:
    """Apply a list of pass names (in order) to the IR string."""
    for pass_name in passes:
        fn = _PASS_FN.get(pass_name)
        if fn is not None:
            ir = fn(ir)
    return ir


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def apply_optimizations(ir: str) -> Tuple[str, Dict[str, Any]]:
    """
    Advisordriven optimization with a safety-verified fallback.

    Pipeline:
      1. Ask the LLM advisor for a suggested pass sequence.
      2. Apply the suggested passes → candidate IR.
      3. Verify equivalence(original IR, candidate IR).
      4a. Verified   → return candidate IR, report decision="accepted".
      4b. Not verified → apply default fixed order → return that IR,
                         report decision="rejected".

    The report dict is the evaluation log.  It records what the LLM suggested,
    what was accepted or rejected, and which passes were ultimately applied.
    This data is the basis for your project's evaluation section.

    Args:
        ir (str): The unoptimized LLVM IR (output of ir.generator.generate_ir).

    Returns:
        Tuple[str, Dict[str, Any]]:
            final_ir   — the optimized (or safely defaulted) LLVM IR string.
            report     — a dict with the following keys:
                "suggested_passes" : List[str]   — what the LLM suggested
                "decision"         : str          — "accepted" or "rejected"
                "rejection_reason" : str | None   — reason if rejected
                "fallback_passes"  : List[str] | None — passes used on fallback
                "final_passes"     : List[str]    — passes actually applied
    """
    report: Dict[str, Any] = {
        "suggested_passes": [],
        "decision": None,
        "rejection_reason": None,
        "fallback_passes": None,
        "final_passes": [],
    }

    # ── Step 1: Ask the advisor ───────────────────────────────────────────────
    suggested = get_optimization_advice(ir)
    report["suggested_passes"] = suggested

    # ── Step 2: Apply suggested passes ───────────────────────────────────────
    try:
        candidate_ir = _apply_pass_sequence(ir, suggested)
    except Exception as e:
        # If applying the suggested passes raises, treat as rejection
        candidate_ir = ir
        suggested_ok = False
        rejection_reason = f"Pass application error: {e}"
    else:
        suggested_ok = True
        rejection_reason = None

    # ── Step 3: Verify equivalence ────────────────────────────────────────────
    if suggested_ok:
        is_equivalent = verify_equivalence(ir, candidate_ir)
    else:
        is_equivalent = False

    # ── Step 4a: Accept ───────────────────────────────────────────────────────
    if is_equivalent:
        report["decision"] = "accepted"
        report["final_passes"] = suggested
        return candidate_ir, report

    # ── Step 4b: Reject → fallback ────────────────────────────────────────────
    if rejection_reason is None:
        rejection_reason = "Output mismatch detected by differential verifier"

    report["decision"] = "rejected"
    report["rejection_reason"] = rejection_reason

    fallback = _default_passes()
    report["fallback_passes"] = fallback

    try:
        fallback_ir = _apply_pass_sequence(ir, fallback)
    except Exception:
        # Even the default order failed — return original IR unchanged
        report["final_passes"] = []
        return ir, report

    report["final_passes"] = fallback
    return fallback_ir, report
