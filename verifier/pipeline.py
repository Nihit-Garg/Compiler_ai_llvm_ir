"""
Fallback Orchestrator for the MiniC compiler.

This module is the integration point between the AI advisor, the optimizer
passes, and the safety verifier.  It implements the core safety guarantee:

    "The AI layer can only make the compiler better — never incorrect."

Public interface:

    apply_optimizations(ir: str, use_advisor: bool = True) -> tuple[str, dict]
        1. Ask the LLM advisor for a suggested pass sequence
           (or take the default order when use_advisor=False).
        2. Apply the suggested passes to produce candidate IR.
        3. Run the differential verifier (original vs candidate).
        4a. If verified → accept, return candidate IR.
        4b. If rejected → apply the fixed default pass order and verify THAT
            against the original as well.
              - verified     → return the default-order IR.
              - not verified → return the original IR unchanged.
        Return the final IR together with a report dict for logging/evaluation.

Nothing leaves this module unless the verifier has passed it or it is the
untouched input.  The default pass order gets no free pass: it is code like
any other and is checked the same way.
"""

from typing import Dict, Any, List, Optional, Tuple

from advisor.heuristics import get_optimization_advice, _default_passes, _VALID_PASSES
from optimizer.passes import (
    constant_folding,
    dead_code_elimination,
    common_subexpression_elimination,
    peephole,
)
from verifier.check import verify_equivalence, equivalence_failure_reason


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

_GENERIC_MISMATCH = "Output mismatch detected by differential verifier"


def _try_passes(ir: str, passes: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Apply a pass sequence and verify the result against the original IR.

    Returns (verified_ir, None) on success, or (None, reason) if the passes
    raised, the verifier raised, or equivalence was not established.
    """
    try:
        candidate_ir = _apply_pass_sequence(ir, passes)
    except Exception as e:
        return None, f"Pass application error: {e}"

    try:
        if verify_equivalence(ir, candidate_ir):
            return candidate_ir, None
        reason = equivalence_failure_reason(ir, candidate_ir)
    except Exception as e:   # a verifier fault must never look like a pass
        return None, f"Verifier error: {type(e).__name__}: {e}"

    return None, reason or _GENERIC_MISMATCH


def apply_optimizations(ir: str, use_advisor: bool = True) -> Tuple[str, Dict[str, Any]]:
    """
    Advisor-driven optimization with a safety-verified fallback.

    Pipeline:
      1. Ask the LLM advisor for a suggested pass sequence.
      2. Apply the suggested passes → candidate IR.
      3. Verify equivalence(original IR, candidate IR).
      4a. Verified     → return candidate IR, report decision="accepted".
      4b. Not verified → report decision="rejected", then apply the default
                         fixed order and verify it too:
            verified     → return that IR      (fallback_decision="accepted")
            not verified → return original IR  (fallback_decision="rejected")

    The report dict is the evaluation log.  It records what the LLM suggested,
    what was accepted or rejected, and which passes were ultimately applied.
    This data is the basis for your project's evaluation section.

    Args:
        ir (str): The unoptimized LLVM IR (output of ir.generator.generate_ir).
        use_advisor (bool): If False, the LLM is not consulted and the default
            pass order is used as the suggestion.  It is still verified: no
            path through this function emits unverified optimized IR.

    Returns:
        Tuple[str, Dict[str, Any]]:
            final_ir   — the optimized IR, or the original IR if nothing
                         could be verified.
            report     — a dict with the following keys:
                "suggested_passes" : List[str]   — what the LLM suggested
                "decision"         : str          — "accepted" or "rejected"
                "rejection_reason" : str | None   — reason if rejected
                "fallback_passes"  : List[str] | None — passes tried on fallback
                "fallback_decision": str | None   — "accepted"/"rejected" for
                                                    the fallback; None if the
                                                    suggestion was accepted
                "fallback_rejection_reason" : str | None
                "final_passes"     : List[str]    — passes actually applied
                                                    ([] = IR left unoptimized)
    """
    report: Dict[str, Any] = {
        "suggested_passes": [],
        "decision": None,
        "rejection_reason": None,
        "fallback_passes": None,
        "fallback_decision": None,
        "fallback_rejection_reason": None,
        "final_passes": [],
    }

    # ── Step 1: Ask the advisor ───────────────────────────────────────────────
    suggested = get_optimization_advice(ir) if use_advisor else _default_passes()
    report["suggested_passes"] = suggested

    # ── Steps 2–3: Apply suggested passes and verify ─────────────────────────
    candidate_ir, rejection_reason = _try_passes(ir, suggested)

    # ── Step 4a: Accept ───────────────────────────────────────────────────────
    if candidate_ir is not None:
        report["decision"] = "accepted"
        report["final_passes"] = suggested
        return candidate_ir, report

    # ── Step 4b: Reject → verified fallback ──────────────────────────────────
    report["decision"] = "rejected"
    report["rejection_reason"] = rejection_reason

    fallback = _default_passes()
    report["fallback_passes"] = fallback

    fallback_ir, fallback_reason = _try_passes(ir, fallback)
    if fallback_ir is not None:
        report["fallback_decision"] = "accepted"
        report["final_passes"] = fallback
        return fallback_ir, report

    # Neither sequence could be verified — ship the original IR untouched.
    report["fallback_decision"] = "rejected"
    report["fallback_rejection_reason"] = fallback_reason
    report["final_passes"] = []
    return ir, report
