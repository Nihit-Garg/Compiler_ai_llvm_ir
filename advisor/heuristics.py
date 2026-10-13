"""
LLM Advisor for the MiniC compiler.

Uses the Gemini API to suggest an ordered sequence of optimization passes
for a given LLVM IR program.

Interface defined in docs/SPEC.md §4:
    get_optimization_advice(ir: str) -> List[str]
    get_advice_with_source(ir: str) -> Tuple[List[str], Dict[str, Any]]

Environment variables
---------------------
GEMINI_API_KEY : str
    # ─────────────────────────────────────────────────────────
    # PLACEHOLDER — set this before running the advisor.
    #
    # Get a free key at: https://aistudio.google.com/apikey
    # Then either:
    #   export GEMINI_API_KEY="your-key-here"      (shell)
    #   or add it to a .env file and load with python-dotenv.
    #
    # If the variable is not set, the advisor silently returns
    # the default fixed pass order so the pipeline keeps working.
    # ─────────────────────────────────────────────────────────

ADVISOR_MODEL : str  (optional)
    Gemini model name to use. Defaults to "gemini-3.5-flash-lite"
    (_DEFAULT_MODEL below).  E.g. export ADVISOR_MODEL="gemini-3.1-flash-lite"
"""

import importlib
import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

# ── Load .env file automatically (local development) ─────────────────────────
# python-dotenv reads .env in the project root and injects variables into
# os.environ.  This is a no-op if the file doesn't exist or if the variable
# is already set (e.g. in CI via GitHub Secrets).
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # looks for .env in cwd and parent dirs
except ImportError:
    pass  # python-dotenv not installed — fall back to plain env vars


# ---------------------------------------------------------------------------
# Valid pass names — must exactly match function names in optimizer/passes.py
# ---------------------------------------------------------------------------

_VALID_PASSES = [
    "constant_folding",
    "dead_code_elimination",
    "common_subexpression_elimination",
    "peephole",
]

_DEFAULT_MODEL = "gemini-3.5-flash-lite"  # GA since 2026-07-21, no shutdown date announced


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(ir: str) -> str:
    """Build the LLM prompt that instructs Gemini to suggest pass ordering."""
    passes_list = "\n".join(f"- {p}" for p in _VALID_PASSES)
    return f"""You are an optimization advisor for a simple compiler.

Given the following LLVM IR program, choose an ordered list of optimization passes to apply.

Available passes (use ONLY these exact names, no others):
{passes_list}

Rules:
- Return ONLY a valid JSON array of pass name strings, e.g. ["peephole", "dead_code_elimination"]
- Each pass may appear at most once.
- Order matters — passes are applied left-to-right.
- If you believe no optimization is beneficial, return an empty array: []
- Do NOT include any explanation, commentary, or text outside the JSON array.

LLVM IR to analyze:
```
{ir.strip()}
```

Your answer (JSON array only):"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(response_text: str) -> List[str]:
    """
    Extract a list of valid pass names from the raw LLM response string.

    Strategy:
    1. Try to find a JSON array in the response.
    2. Filter the parsed list to only include known-valid pass names.
    3. Deduplicate while preserving order.
    Returns an empty list on any parse failure, so it cannot tell an explicit
    "[]" from garbage; use _parse_response_strict() for that.
    """
    return _parse_response_strict(response_text) or []


def _parse_response_strict(response_text: str) -> Optional[List[str]]:
    """
    Like _parse_response(), but separates the two meanings of an empty result:

    - []    the LLM answered with an explicit empty array ("no optimization")
    - None  the answer is unusable: no JSON array, malformed JSON, or a
            non-empty array without a single valid pass name
    """
    # Attempt to extract a JSON array from the text
    match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, list):
        return None

    if not parsed:
        return []

    # Keep only valid, non-duplicate pass names (in the order the LLM gave them)
    seen = set()
    result: List[str] = []
    for item in parsed:
        if isinstance(item, str) and item in _VALID_PASSES and item not in seen:
            seen.add(item)
            result.append(item)

    return result or None


# ---------------------------------------------------------------------------
# Default pass order (fallback)
# ---------------------------------------------------------------------------

def _default_passes() -> List[str]:
    """Return the standard fixed optimization pass order."""
    return list(_VALID_PASSES)  # constant_folding → DCE → CSE → peephole


# ---------------------------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, api_key: str, model: str) -> Optional[str]:
    """
    Call the Gemini API and return the text of the first response candidate.
    Uses the new `google-genai` SDK (google.genai).
    Returns None on any error (network, auth, quota, etc.).
    """
    try:
        from google import genai  # type: ignore
    except ImportError:
        # google-genai not installed — fall back silently
        return None

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text
    except Exception:
        # Covers auth errors, network issues, quota exhaustion, etc.
        return None


def _genai_installed() -> bool:
    """True if the google-genai SDK can be imported."""
    try:
        importlib.import_module("google.genai")
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Public interface (SPEC.md §4)
# ---------------------------------------------------------------------------

# Values of the "source" field returned by get_advice_with_source().
SOURCE_LLM               = "llm"                # the LLM's own answer is used
SOURCE_NO_API_KEY        = "no_api_key"         # GEMINI_API_KEY unset or blank
SOURCE_SDK_UNAVAILABLE   = "sdk_unavailable"    # google-genai not installed
SOURCE_API_ERROR         = "api_error"          # the API call failed
SOURCE_UNUSABLE_RESPONSE = "unusable_response"  # no valid pass list in the answer
SOURCE_SKIPPED           = "skipped"            # advisor not consulted (set by the pipeline)


def get_advice_with_source(ir: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Same advice as get_optimization_advice(), plus a record of where it came
    from, so a report can tell a real LLM answer from the default fallback.

    Returns (passes, info) where info has:
        "source" : str        — one of the SOURCE_* values above
        "model"  : str | None — model name, if an API call was attempted

    An explicit empty answer ("[]") from the LLM is honoured: passes is []
    and source is "llm".  Every other failure returns the default order.
    """
    # ── Check for API key ────────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key:
        # No key → silent fallback (offline / CI mode)
        return _default_passes(), {"source": SOURCE_NO_API_KEY, "model": None}

    model = os.environ.get("ADVISOR_MODEL", _DEFAULT_MODEL).strip()

    # ── Build prompt and call Gemini ─────────────────────────────────────────
    prompt = _build_prompt(ir)
    raw_response = _call_gemini(prompt, api_key, model)

    if raw_response is None:
        # API call failed → silent fallback
        source = SOURCE_API_ERROR if _genai_installed() else SOURCE_SDK_UNAVAILABLE
        return _default_passes(), {"source": source, "model": model}

    # ── Parse response ───────────────────────────────────────────────────────
    passes = _parse_response_strict(raw_response)

    if passes is None:
        # LLM gave no usable output → fall back
        return _default_passes(), {"source": SOURCE_UNUSABLE_RESPONSE, "model": model}

    return passes, {"source": SOURCE_LLM, "model": model}


def get_optimization_advice(ir: str) -> List[str]:
    """
    Analyzes the given LLVM IR and returns an ordered list of suggested
    optimization pass names to apply.

    Uses the Gemini API if GEMINI_API_KEY is set in the environment.
    Falls back to the default fixed pass order silently on any failure:
      - API key not set
      - Network / auth error
      - Unparseable response
      - Response contains no valid pass names

    Args:
        ir (str): The LLVM IR text to analyze.

    Returns:
        List[str]: Ordered list of pass names from _VALID_PASSES.
                   The default order on any failure.  Empty only if the LLM
                   explicitly answered [] ("no optimization is beneficial").
    """
    passes, _ = get_advice_with_source(ir)
    return passes
