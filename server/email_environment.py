"""
Main environment class: implements reset(), step(), and state().

This is a standard Gymnasium-style interface wrapped for OpenEnv.
State is held in memory (single-session); no database required.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from models import (
    EmailAction,
    EmailObservation,
    EmailReward,
    EmailState,
    InboxEmail,
)
from server.graders import BILLING_DISPUTE_RUBRIC, grade_classification, grade_prioritization, grade_reply
from server.tasks import TASK_REGISTRY, VALID_TASK_IDS, TaskDefinition

_DATA_FILE = Path(__file__).parent / "data" / "emails.json"


def _load_emails() -> dict[str, dict]:
    """Load the email dataset once at import time."""
    with open(_DATA_FILE, encoding="utf-8") as f:
        emails = json.load(f)
    return {e["id"]: e for e in emails}


_EMAIL_DB: dict[str, dict] = _load_emails()


class EmailEnvironment:
    """
    Email Triage Environment.

    Episode flow:
      1. Call reset(task_id) → returns first EmailObservation
      2. Repeatedly call step(action) → returns (EmailObservation, EmailReward, done, info)
      3. When done=True, episode is finished; call reset() to start a new one
    """

    def __init__(self) -> None:
        self._state: EmailState | None = None
        self._task: TaskDefinition | None = None
        self._email_queue: list[dict] = []
        self._action_history: list[str] = []  # For repeated-action detection

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task_id: str | None = None) -> EmailObservation:
        """
        Start a new episode.

        Args:
            task_id: One of 'classify', 'prioritize', 'reply'.
                     If None, defaults to 'classify'.

        Returns:
            First observation of the episode.
        """
        if task_id is None:
            task_id = "classify"

        if task_id not in VALID_TASK_IDS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {VALID_TASK_IDS}")

        self._task = TASK_REGISTRY[task_id]
        self._email_queue = [_EMAIL_DB[eid] for eid in self._task.email_ids if eid in _EMAIL_DB]
        self._action_history = []

        self._state = EmailState(
            task_id=task_id,
            current_step=0,
            max_steps=self._task.max_steps,
            emails_processed=[],
            running_score=0.0,
            step_scores=[],
            history=[],
            is_done=False,
            episode_id=str(uuid.uuid4()),
        )

        return self._build_observation()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: EmailAction) -> tuple[EmailObservation, EmailReward, bool, dict[str, Any]]:
        """
        Execute one action in the environment.

        Returns:
            (observation, reward, done, info)
        """
        if self._state is None or self._task is None:
            raise RuntimeError("Call reset() before step().")

        if self._state.is_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        info: dict[str, Any] = {}

        # ---- Validate action type ----
        if action.action_type != self._task.required_action_type:
            reward = EmailReward(
                score=0.01,
                feedback=(
                    f"Wrong action type '{action.action_type}'. "
                    f"Task '{self._task.task_id}' requires '{self._task.required_action_type}'."
                ),
                done=False,
            )
            info["error"] = "wrong_action_type"
            # Still advance step counter to prevent gaming
            self._state.current_step += 1
            if self._state.current_step >= self._state.max_steps:
                self._state.is_done = True
                reward.done = True
                reward.episode_score = self._compute_episode_score()
            return self._build_observation(), reward, self._state.is_done, info

        # ---- Repeated-action penalty ----
        action_signature = self._action_signature(action)
        if action_signature in self._action_history:
            reward = EmailReward(
                score=0.01,
                feedback="Penalty: identical action submitted twice. Try a different approach.",
                done=False,
            )
            info["warning"] = "repeated_action"
            self._state.current_step += 1
            if self._state.current_step >= self._state.max_steps:
                self._state.is_done = True
                reward.done = True
                reward.episode_score = self._compute_episode_score()
            return self._build_observation(), reward, self._state.is_done, info

        self._action_history.append(action_signature)

        # ---- Grade the action ----
        grade_result = self._grade_action(action)
        step_score = grade_result["score"]
        self._state.step_scores.append(step_score)

        # Record to history
        current_email_id = self._current_email_id()
        self._state.history.append({
            "step": self._state.current_step,
            "email_id": current_email_id,
            "action_type": action.action_type,
            "score": step_score,
            "feedback": grade_result.get("feedback", ""),
        })

        # Mark email as processed
        if current_email_id:
            self._state.emails_processed.append(current_email_id)

        # Update step counter
        self._state.current_step += 1

        # ---- Check done conditions ----
        emails_remaining = len(self._email_queue) - self._state.current_step
        step_limit_hit = self._state.current_step >= self._state.max_steps

        done = step_limit_hit or emails_remaining <= 0

        episode_score: float | None = None
        if done:
            self._state.is_done = True
            self._state.running_score = self._compute_episode_score()
            episode_score = self._state.running_score

        reward = EmailReward(
            score=step_score,
            partial_scores=grade_result,
            feedback=grade_result.get("feedback", ""),
            done=done,
            episode_score=episode_score,
        )

        return self._build_observation(), reward, done, info

    # ------------------------------------------------------------------
    # state (property)
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        """Return full serializable snapshot of current environment state."""
        if self._state is None:
            return {"status": "not_started"}
        return self._state.model_dump()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_email_id(self) -> str | None:
        idx = self._state.current_step if self._state else 0
        if idx < len(self._email_queue):
            return self._email_queue[idx].get("id")
        return None

    def _build_observation(self) -> EmailObservation:
        """Build the observation for the current state."""
        assert self._state is not None
        assert self._task is not None

        step = self._state.current_step

        if self._state.is_done:
            return EmailObservation(
                task_id=self._state.task_id,
                step_number=step,
                is_done=True,
                task_description="Episode complete.",
            )

        # For prioritize task: show all emails at once as inbox_context
        if self._task.task_id == "prioritize":
            inbox_context = [
                InboxEmail(
                    email_id=e["id"],
                    subject=e["subject"],
                    sender=e["sender"],
                    timestamp=e["timestamp"],
                )
                for e in self._email_queue
            ]
            return EmailObservation(
                task_id=self._state.task_id,
                step_number=step,
                inbox_context=inbox_context,
                task_description=self._task.description,
                is_done=False,
            )

        # For classify and reply: show one email at a time
        if step < len(self._email_queue):
            email = self._email_queue[step]
            return EmailObservation(
                task_id=self._state.task_id,
                step_number=step,
                email_id=email.get("id"),
                subject=email.get("subject"),
                sender=email.get("sender"),
                timestamp=email.get("timestamp"),
                body=email.get("body"),
                thread_history=email.get("thread_history", []),
                task_description=self._task.description,
                is_done=False,
            )

        # No more emails but episode not marked done yet (edge case)
        return EmailObservation(
            task_id=self._state.task_id,
            step_number=step,
            is_done=True,
            task_description="All emails processed.",
        )

    def _grade_action(self, action: EmailAction) -> dict[str, Any]:
        """Route the action to the appropriate grader."""
        task_id = self._task.task_id  # type: ignore[union-attr]

        if task_id == "classify":
            step = self._state.current_step  # type: ignore[union-attr]
            if step < len(self._email_queue):
                email = self._email_queue[step]
                ground_truth_label = self._task.ground_truth.get(email["id"], "normal")  # type: ignore[union-attr]
                return grade_classification(action, ground_truth_label)
            return {"score": 0.01, "feedback": "No email to classify."}

        elif task_id == "prioritize":
            return grade_prioritization(action, self._task.ground_truth)  # type: ignore[union-attr]

        elif task_id == "reply":
            return grade_reply(action, BILLING_DISPUTE_RUBRIC)

        return {"score": 0.01, "feedback": f"Unknown task '{task_id}'."}

    def _compute_episode_score(self) -> float:
        """Compute the final episode score as mean of step scores."""
        if not self._state or not self._state.step_scores:
            return 0.01
        return round(sum(self._state.step_scores) / len(self._state.step_scores), 4)

    @staticmethod
    def _action_signature(action: EmailAction) -> str:
        """Create a string signature for repeated-action detection."""
        return (
            f"{action.action_type}|"
            f"{action.label}|"
            f"{','.join(action.priority_order or [])}|"
            f"{(action.reply_text or '')[:50]}"
        )
