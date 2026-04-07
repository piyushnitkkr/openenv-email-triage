"""
OpenEnv Email Triage — Inference Script
========================================
Required environment variables:
  API_BASE_URL  - Base URL of the LLM API (OpenAI-compatible endpoint)
  MODEL_NAME    - Model identifier for inference (e.g. "gpt-4o")
  HF_TOKEN      - Hugging Face / API key used for authentication

The environment server is expected to be running on http://localhost:7860
(when deployed on Hugging Face Spaces).

Usage:
    API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o HF_TOKEN=sk-... python inference.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from typing import Any

import httpx

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration from environment variables (required by submission spec)
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SEED = int(os.environ.get("SEED", "42"))

# The OpenEnv environment server base URL (self-referential when running inside HF Space)
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable is not set.")
    sys.exit(1)

random.seed(SEED)

# ---------------------------------------------------------------------------
# OpenAI client — uses API_BASE_URL + HF_TOKEN as required by the spec
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# Environment interaction helpers
# ---------------------------------------------------------------------------


def reset_env(task_id: str) -> dict:
    r = httpx.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def step_env(action: dict) -> dict:
    r = httpx.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def get_tasks() -> list[dict]:
    r = httpx.get(f"{ENV_BASE_URL}/tasks", timeout=30)
    r.raise_for_status()
    return r.json()["tasks"]


# ---------------------------------------------------------------------------
# Agent logic
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert email triage agent. You will receive email content and must respond
with a JSON action object. Always output ONLY valid JSON — no markdown, no commentary.

Action formats:
  Classify:   {"action_type": "classify", "label": "<spam|urgent|normal|newsletter>", "reasoning": "..."}
  Prioritize: {"action_type": "prioritize", "priority_order": ["email_id1", "email_id2", ...], "reasoning": "..."}
  Reply:      {"action_type": "reply", "reply_text": "...", "reasoning": "..."}
"""


def call_agent(user_prompt: str) -> dict[str, Any] | None:
    """Call the LLM via OpenAI-compatible client and parse the JSON action."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            seed=SEED,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  [!] Failed to parse agent response as JSON: {e}")
        return None
    except Exception as e:
        print(f"  [!] LLM API error: {e}")
        return None


def build_user_prompt(obs: dict, task_description: str) -> str:
    """Format an observation into a user prompt."""
    parts = [f"TASK: {task_description}\n"]

    if obs.get("inbox_context"):
        parts.append("INBOX (all emails):")
        for email in obs["inbox_context"]:
            parts.append(
                f"  - ID: {email['email_id']} | Subject: {email['subject']} | "
                f"From: {email['sender']} | {email['timestamp']}"
            )
        parts.append("\nRank ALL emails from most urgent to least urgent.")

    elif obs.get("body"):
        parts.append(f"From: {obs.get('sender', 'unknown')}")
        parts.append(f"Subject: {obs.get('subject', '(no subject)')}")
        parts.append(f"Timestamp: {obs.get('timestamp', '')}")
        if obs.get("thread_history"):
            parts.append("\n--- Thread History (oldest first) ---")
            for msg in obs["thread_history"]:
                parts.append(msg)
            parts.append("--- End Thread ---")
        parts.append(f"\nEmail Body:\n{obs['body']}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(task_id: str, task_description: str) -> dict[str, Any]:
    """Run a full episode for a given task. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_id.upper()}")
    print(f"{'='*60}")
    print(f"[START] task={task_id}", flush=True)

    obs = reset_env(task_id)
    step_scores: list[float] = []
    total_steps = 0
    done = False

    while not done:
        if obs.get("is_done"):
            break

        prompt = build_user_prompt(obs, task_description)
        print(f"\n  Step {total_steps + 1}: calling agent...")

        action = call_agent(prompt)
        if action is None:
            action = {"action_type": "classify", "label": "normal"}

        # Truncate reply_text in log output for readability
        log_action = {k: v for k, v in action.items() if k != "reply_text"}
        print(f"  Action: {json.dumps(log_action, ensure_ascii=False)[:120]}")

        result = step_env(action)
        reward = result.get("reward", {})
        score = reward.get("score", 0.0)
        feedback = reward.get("feedback", "")
        done = result.get("done", False)
        obs = result.get("observation", {})

        step_scores.append(score)
        total_steps += 1
        print(f"  Score: {score:.3f} | Feedback: {feedback}")
        print(f"[STEP] step={total_steps} reward={score}", flush=True)

    episode_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
    print(f"[END] task={task_id} score={episode_score} steps={total_steps}", flush=True)
    return {
        "task_id": task_id,
        "difficulty": next((t.get("difficulty", "?") for t in tasks if t["id"] == task_id), "?"),
        "steps": total_steps,
        "episode_score": episode_score,
        "step_scores": step_scores,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nOpenEnv Email Triage — Inference")
    print(f"Model     : {MODEL_NAME}")
    print(f"API Base  : {API_BASE_URL}")
    print(f"Env URL   : {ENV_BASE_URL}")
    print(f"Seed      : {SEED}")

    # Verify server is up
    try:
        status = httpx.get(f"{ENV_BASE_URL}/", timeout=10).json()
        print(f"Server status: {status}")
    except Exception as e:
        print(f"ERROR: Cannot reach environment server at {ENV_BASE_URL}: {e}")
        sys.exit(1)

    tasks = get_tasks()
    task_map = {t["id"]: t.get("description", t["id"]) for t in tasks}

    results: list[dict] = []
    for task in tasks:
        result = run_episode(task["id"], task_map[task["id"]])
        results.append(result)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  RESULTS  (model={MODEL_NAME}, seed={SEED})")
    print(f"{'='*60}")
    print(f"  {'Task':<14} {'Difficulty':<12} {'Steps':<8} {'Score':<10} {'Expected Range'}")
    print(f"  {'-'*57}")
    expected = {"classify": "0.75 – 0.90", "prioritize": "0.45 – 0.65", "reply": "0.25 – 0.50"}
    for r in results:
        print(
            f"  {r['task_id']:<14} {r['difficulty']:<12} {r['steps']:<8} "
            f"{r['episode_score']:.4f}     {expected.get(r['task_id'], '?')}"
        )

    mean_score = sum(r["episode_score"] for r in results) / len(results)
    print(f"\n  Overall mean: {mean_score:.4f}")
    print(f"{'='*60}\n")

    # Save results as JSON for the /baseline API endpoint to read
    with open("baseline_results.json", "w") as f:
        json.dump({"scores": results, "mean_score": mean_score}, f)

    print("Results saved to baseline_results.json")
