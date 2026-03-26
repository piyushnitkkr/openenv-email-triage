"""
Task definitions for the 3 email triage tasks.

Each task specifies which emails to use, what action is required,
and the ground truth for grading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskDefinition:
    """Describes a single task episode."""
    task_id: str
    description: str
    difficulty: str
    email_ids: list[str]
    required_action_type: str
    max_steps: int
    action_description: str
    ground_truth: Any  # label per email_id (dict) or ordered list or rubric dict


# ---------------------------------------------------------------------------
# Task 1 — Email Classification (Easy)
# ---------------------------------------------------------------------------
# 10 emails spanning all 4 categories; agent must classify each one.

TASK_CLASSIFY = TaskDefinition(
    task_id="classify",
    description=(
        "You will receive 10 emails one at a time. "
        "For each email, classify it into exactly one of these categories: "
        "spam, urgent, normal, newsletter. "
        "Submit a 'classify' action with the 'label' field set."
    ),
    difficulty="easy",
    # Mix: 2 spam, 3 urgent, 3 normal, 2 newsletter
    email_ids=[
        "email_001",  # spam
        "email_002",  # urgent
        "email_003",  # newsletter
        "email_004",  # urgent
        "email_005",  # normal
        "email_006",  # urgent
        "email_007",  # newsletter
        "email_008",  # normal
        "email_009",  # urgent  (note: reclassified for task balance)
        "email_012",  # spam
    ],
    required_action_type="classify",
    max_steps=10,
    action_description="classify with label: spam | urgent | normal | newsletter",
    # email_id → correct label
    ground_truth={
        "email_001": "spam",
        "email_002": "urgent",
        "email_003": "newsletter",
        "email_004": "urgent",
        "email_005": "normal",
        "email_006": "urgent",
        "email_007": "newsletter",
        "email_008": "normal",
        "email_009": "urgent",
        "email_012": "spam",
    },
)

# ---------------------------------------------------------------------------
# Task 2 — Inbox Prioritization (Medium)
# ---------------------------------------------------------------------------
# 10 emails shown together; agent ranks them from most to least urgent.

TASK_PRIORITIZE = TaskDefinition(
    task_id="prioritize",
    description=(
        "You will see 10 emails in your inbox at once. "
        "Rank them from most urgent to least urgent. "
        "Submit a 'prioritize' action with 'priority_order' as a list of email_ids "
        "ordered from highest priority (#1) to lowest priority (#10)."
    ),
    difficulty="medium",
    email_ids=[
        "email_040",  # P1 payment outage          → priority 1
        "email_002",  # production server down      → priority 2
        "email_034",  # AWS bill +348%              → priority 3
        "email_021",  # CEO needs board materials   → priority 4
        "email_009",  # mobile Safari crash + demo  → priority 5
        "email_031",  # contract expiring Jan 31    → priority 6
        "email_013",  # DB migration Saturday       → priority 7
        "email_008",  # performance review - HR     → priority 8
        "email_025",  # expense report approval     → priority 9
        "email_026",  # birthday celebration        → priority 10
    ],
    required_action_type="prioritize",
    max_steps=1,
    action_description="prioritize with priority_order: [email_id, ...] most to least urgent",
    # Correct ordering (most → least urgent)
    ground_truth=[
        "email_040",
        "email_002",
        "email_034",
        "email_021",
        "email_009",
        "email_031",
        "email_013",
        "email_008",
        "email_025",
        "email_026",
    ],
)

# ---------------------------------------------------------------------------
# Task 3 — Reply Drafting (Hard)
# ---------------------------------------------------------------------------
# A multi-turn billing dispute thread; agent drafts a professional reply.

TASK_REPLY = TaskDefinition(
    task_id="reply",
    description=(
        "You are a customer support representative. "
        "Read the billing dispute thread below carefully and draft a professional reply. "
        "Your reply must: acknowledge the customer's frustration, reference the invoice, "
        "commit to a refund investigation with a concrete timeline, and close professionally. "
        "Do NOT invent facts not present in the thread. "
        "Submit a 'reply' action with 'reply_text' containing your full response."
    ),
    difficulty="hard",
    email_ids=["email_029"],  # Sarah Chen's final escalation
    required_action_type="reply",
    max_steps=1,
    action_description="reply with reply_text: your drafted response",
    # The rubric is used by graders.py (BILLING_DISPUTE_RUBRIC)
    ground_truth=None,  # grader uses its built-in rubric
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, TaskDefinition] = {
    "classify": TASK_CLASSIFY,
    "prioritize": TASK_PRIORITIZE,
    "reply": TASK_REPLY,
}

VALID_TASK_IDS = list(TASK_REGISTRY.keys())
