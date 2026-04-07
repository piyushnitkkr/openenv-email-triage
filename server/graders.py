"""
Deterministic grader functions for all 3 tasks.

All graders return a float in [0.0, 1.0].
Same inputs always produce the same output (no randomness, no LLM calls).
"""
from __future__ import annotations

from typing import Any

from models import EmailAction


# ---------------------------------------------------------------------------
# Task 1: Email Classification
# ---------------------------------------------------------------------------

# Near-miss pairs: classifying as one of these when the other is correct
# earns partial credit because the mistake is understandable.
_NEAR_MISS_PAIRS: set[frozenset[str]] = {
    frozenset({"urgent", "normal"}),
    frozenset({"normal", "newsletter"}),
}


def grade_classification(action: EmailAction, ground_truth_label: str) -> dict[str, Any]:
    """
    Grade a classify action.

    Returns:
        dict with keys: score (float), feedback (str), exact_match (bool)
    """
    if not action.label:
        return {
            "score": 0.01,
            "feedback": "No label provided in action.",
            "exact_match": False,
        }

    predicted = action.label.lower().strip()
    truth = ground_truth_label.lower().strip()

    if predicted == truth:
        return {
            "score": 0.99,
            "feedback": f"Correct! Label '{predicted}' matches ground truth.",
            "exact_match": True,
        }

    if frozenset({predicted, truth}) in _NEAR_MISS_PAIRS:
        return {
            "score": 0.5,
            "feedback": f"Partial credit: predicted '{predicted}', ground truth is '{truth}' (near-miss).",
            "exact_match": False,
        }

    return {
        "score": 0.01,
        "feedback": f"Incorrect: predicted '{predicted}', ground truth is '{truth}'.",
        "exact_match": False,
    }


# ---------------------------------------------------------------------------
# Task 2: Inbox Prioritization
# ---------------------------------------------------------------------------

def _kendall_tau_distance(predicted: list[str], ground_truth: list[str]) -> float:
    """
    Compute normalized Kendall tau distance between two orderings.

    Returns a value in [0.0, 1.0] where:
      0.0 = identical ordering (maximum discordance ... wait, we INVERT: 1.0 = perfect)
      1.0 = perfect agreement

    We count concordant vs discordant pairs.
    """
    n = len(predicted)
    if n <= 1:
        return 1.0

    # Build index map: email_id → rank in each list
    pred_rank = {email_id: i for i, email_id in enumerate(predicted)}
    truth_rank = {email_id: i for i, email_id in enumerate(ground_truth)}

    # Only score emails present in both lists
    common = [eid for eid in ground_truth if eid in pred_rank]
    n = len(common)
    if n <= 1:
        return 1.0

    concordant = 0
    discordant = 0
    total_pairs = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            ei, ej = common[i], common[j]
            pred_order = pred_rank[ei] < pred_rank[ej]
            truth_order = truth_rank[ei] < truth_rank[ej]
            if pred_order == truth_order:
                concordant += 1
            else:
                discordant += 1

    if total_pairs == 0:
        return 1.0

    # Normalized: 1.0 = perfect agreement, 0.0 = perfect inversion
    return (concordant - discordant) / (2 * total_pairs) + 0.5


def grade_prioritization(action: EmailAction, ground_truth_order: list[str]) -> dict[str, Any]:
    """
    Grade a prioritize action using normalized Kendall tau distance.
    Top-3 positions are weighted 2x.

    Returns:
        dict with keys: score (float), feedback (str), tau (float), top3_score (float)
    """
    if not action.priority_order:
        return {
            "score": 0.01,
            "feedback": "No priority_order provided in action.",
            "tau": 0.0,
            "top3_score": 0.0,
        }

    predicted = action.priority_order
    truth = ground_truth_order

    # Full list Kendall tau
    tau = _kendall_tau_distance(predicted, truth)

    # Top-3 bonus: what fraction of top-3 truth emails appear in predicted top-3?
    top3_truth = set(truth[:3])
    top3_pred = set(predicted[:3])
    top3_overlap = len(top3_truth & top3_pred) / 3.0

    # Weighted combination: 60% overall tau, 40% top-3 overlap
    score = 0.6 * tau + 0.4 * top3_overlap
    score = max(0.01, min(0.99, score))

    return {
        "score": round(score, 4),
        "feedback": (
            f"Kendall tau={tau:.3f}, top-3 overlap={top3_overlap:.2f}. "
            f"Combined score={score:.3f}."
        ),
        "tau": round(tau, 4),
        "top3_score": round(top3_overlap, 4),
    }


# ---------------------------------------------------------------------------
# Task 3: Reply Drafting
# ---------------------------------------------------------------------------

# Rubric for the billing dispute reply (email_029 / email_004 / email_014)
BILLING_DISPUTE_RUBRIC: dict[str, dict[str, Any]] = {
    "acknowledgment": {
        "description": "Acknowledges the dispute and the customer's frustration",
        "keywords": [
            "apologize", "sorry", "understand", "acknowledge",
            "frustration", "inconvenience", "regret"
        ],
        "weight": 0.20,
    },
    "invoice_reference": {
        "description": "Mentions the invoice number INV-2024-0089 or the amount $2,450",
        "keywords": ["inv-2024-0089", "2,450", "2450", "invoice"],
        "weight": 0.15,
    },
    "refund_commitment": {
        "description": "Commits to processing or investigating the refund",
        "keywords": [
            "refund", "credit", "reimburse", "return", "process",
            "investigate", "review"
        ],
        "weight": 0.25,
    },
    "timeline": {
        "description": "Provides a concrete resolution timeline",
        "keywords": [
            "business day", "within", "hours", "days",
            "by end of", "today", "immediately", "expedite"
        ],
        "weight": 0.20,
    },
    "professional_close": {
        "description": "Ends professionally (sign-off, contact info, or gratitude)",
        "keywords": [
            "sincerely", "regards", "thank you", "please contact",
            "reach out", "feel free", "best"
        ],
        "weight": 0.10,
    },
    "no_wrong_facts": {
        "description": "Does not hallucinate amounts, names, or facts not in thread",
        # This is a PENALTY check: if wrong facts found, score is 0 for this criterion
        "penalty_triggers": [
            "1,000", "3,450", "4,450", "jan wright", "james chen",  # wrong names/amounts
        ],
        "weight": 0.10,
    },
}


def grade_reply(action: EmailAction, rubric: dict[str, dict] | None = None) -> dict[str, Any]:
    """
    Grade a reply action against the billing dispute rubric.

    Returns:
        dict with keys: score (float), feedback (str), criteria_scores (dict)
    """
    if rubric is None:
        rubric = BILLING_DISPUTE_RUBRIC

    if not action.reply_text:
        return {
            "score": 0.01,
            "feedback": "No reply_text provided in action.",
            "criteria_scores": {},
        }

    text_lower = action.reply_text.lower()
    criteria_scores: dict[str, float] = {}
    feedback_parts: list[str] = []
    total_score = 0.0

    for criterion, config in rubric.items():
        weight = config["weight"]

        if criterion == "no_wrong_facts":
            # Penalty check: if any penalty trigger word found → 0
            triggered = [kw for kw in config.get("penalty_triggers", []) if kw in text_lower]
            if triggered:
                criteria_scores[criterion] = 0.0
                feedback_parts.append(
                    f"PENALTY [{criterion}]: hallucinated content detected ({triggered})"
                )
            else:
                criteria_scores[criterion] = 1.0
                total_score += weight
        else:
            keywords = config.get("keywords", [])
            matched = [kw for kw in keywords if kw in text_lower]
            if matched:
                criteria_scores[criterion] = 1.0
                total_score += weight
                feedback_parts.append(f"✓ [{criterion}]: matched {matched[:2]}")
            else:
                criteria_scores[criterion] = 0.0
                feedback_parts.append(f"✗ [{criterion}]: no keywords found (expected one of {keywords[:3]})")

    score = max(0.01, min(0.99, total_score))

    return {
        "score": round(score, 4),
        "feedback": " | ".join(feedback_parts),
        "criteria_scores": criteria_scores,
    }
