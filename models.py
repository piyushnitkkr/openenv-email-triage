"""
Pydantic models for the Email Triage OpenEnv environment.

These define the Action, Observation, and State contracts used by both
the server (environment logic) and the client (agent-facing API).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class EmailAction(BaseModel):
    """Action submitted by the agent."""

    action_type: Literal["classify", "prioritize", "reply"] = Field(
        description="Type of action to perform"
    )
    # For classify
    label: Optional[Literal["spam", "urgent", "normal", "newsletter"]] = Field(
        default=None,
        description="Email category label (required for classify action)",
    )
    # For prioritize
    priority_order: Optional[list[str]] = Field(
        default=None,
        description="Ordered list of email_ids from most to least urgent (required for prioritize)",
    )
    # For reply
    reply_text: Optional[str] = Field(
        default=None,
        description="Drafted reply text (required for reply action)",
    )
    # Optional
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning (optional, not scored)",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class InboxEmail(BaseModel):
    """A brief summary of an email shown in inbox context."""
    email_id: str
    subject: str
    sender: str
    timestamp: str


class EmailObservation(BaseModel):
    """Observation returned to the agent after reset() or step()."""

    task_id: str = Field(description="Active task identifier")
    step_number: int = Field(description="Current step within the episode")
    email_id: Optional[str] = Field(default=None, description="ID of the current email")
    subject: Optional[str] = Field(default=None)
    sender: Optional[str] = Field(default=None)
    timestamp: Optional[str] = Field(default=None)
    body: Optional[str] = Field(default=None)
    thread_history: list[str] = Field(
        default_factory=list,
        description="Prior messages in the thread (oldest first)",
    )
    inbox_context: list[InboxEmail] = Field(
        default_factory=list,
        description="Brief summaries of all emails (used in prioritize task)",
    )
    task_description: Optional[str] = Field(
        default=None,
        description="Human-readable description of what the agent must do",
    )
    is_done: bool = Field(default=False, description="Whether the episode is complete")


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class EmailReward(BaseModel):
    """Reward returned after each step()."""

    score: float = Field(ge=0.0, le=1.0, description="Step reward in [0.0, 1.0]")
    partial_scores: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-criterion breakdown",
    )
    feedback: str = Field(default="", description="Human-readable feedback")
    done: bool = Field(default=False, description="Whether the episode is complete")
    episode_score: Optional[float] = Field(
        default=None,
        description="Final aggregated episode score (only set when done=True)",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class EmailState(BaseModel):
    """Internal serializable state snapshot."""

    task_id: str
    current_step: int = 0
    max_steps: int = 10
    emails_processed: list[str] = Field(default_factory=list)
    running_score: float = 0.0
    step_scores: list[float] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    is_done: bool = False
    episode_id: Optional[str] = None
