"""
Baseline inference script using the OpenAI API.

Runs all 3 tasks and prints a results table.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline/run_baseline.py [--seed 42] [--model gpt-4o] [--base-url http://localhost:8000]
"""
from __future__ import annotations

import argparse
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
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="OpenEnv Email Triage Baseline Agent")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
parser.add_argument(
    "--base-url",
    default="http://localhost:8000",
    help="Base URL of the running environment server",
)
args = parser.parse_args()

random.seed(args.seed)

# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------


def reset_env(task_id: str) -> dict:
    r = httpx.post(f"{args.base_url}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def step_env(action: dict) -> dict:
    r = httpx.post(f"{args.base_url}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def get_tasks() -> list[dict]:
    r = httpx.get(f"{args.base_url}/tasks", timeout=30)
    r.raise_for_status()
    return r.json()["tasks"]


# ---------------------------------------------------------------------------
# OpenAI agent
# ---------------------------------------------------------------------------

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

SYSTEM_PROMPT = """\
You are an expert email triage agent. You will receive email content and must respond
with a JSON action object. Always output ONLY valid JSON — no markdown, no commentary.

Action formats:
  Classify:   {"action_type": "classify", "label": "<spam|urgent|normal|newsletter>", "reasoning": "..."}
  Prioritize: {"action_type": "prioritize", "priority_order": ["email_id1", "email_id2", ...], "reasoning": "..."}
  Reply:      {"action_type": "reply", "reply_text": "...", "reasoning": "..."}
"""


def call_agent(user_prompt: str) -> dict[str, Any] | None:
    """Call the OpenAI API and parse the JSON action response."""
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            seed=args.seed,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  [!] Failed to parse agent response as JSON: {e}")
        return None
    except Exception as e:
        print(f"  [!] OpenAI API error: {e}")
        return None


def build_user_prompt(obs: dict, task_description: str) -> str:
    """Format an observation into a user prompt."""
    parts = [f"TASK: {task_description}\n"]

    if obs.get("inbox_context"):
        parts.append("INBOX (all emails):")
        for email in obs["inbox_context"]:
            parts.append(
                f"  - ID: {email['email_id']} | Subject: {email['subject']} | From: {email['sender']} | {email['timestamp']}"
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
    """Run a full episode for a given task."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_id.upper()}")
    print(f"{'='*60}")

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
            # Fallback action if agent fails
            action = {"action_type": "classify", "label": "normal"}

        print(f"  Action: {json.dumps({k: v for k, v in action.items() if k != 'reply_text'}, ensure_ascii=False)[:120]}")

        result = step_env(action)
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]

        score = reward.get("score", 0.0)
        step_scores.append(score)
        total_steps += 1

        print(f"  Score: {score:.3f} | Feedback: {reward.get('feedback', '')[:80]}")

        if done or obs.get("is_done"):
            break

    episode_score = reward.get("episode_score") or (
        sum(step_scores) / len(step_scores) if step_scores else 0.0
    )

    return {
        "task_id": task_id,
        "steps": total_steps,
        "step_scores": step_scores,
        "episode_score": round(episode_score, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"\nOpenEnv Email Triage Baseline")
    print(f"Model: {args.model} | Seed: {args.seed}")
    print(f"Server: {args.base_url}")

    # Check server health
    try:
        r = httpx.get(f"{args.base_url}/", timeout=5)
        r.raise_for_status()
        print(f"Server status: {r.json()}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.base_url}: {e}")
        sys.exit(1)

    task_info = {t["id"]: t["description"] for t in get_tasks()}

    results = []
    for task_id in ["classify", "prioritize", "reply"]:
        result = run_episode(task_id, task_info.get(task_id, ""))
        results.append(result)

    # Print results table
    print(f"\n{'='*60}")
    print(f"  BASELINE RESULTS  (model={args.model}, seed={args.seed})")
    print(f"{'='*60}")
    print(f"  {'Task':<12} {'Difficulty':<12} {'Steps':<8} {'Score':<10} {'Expected Range'}")
    print(f"  {'-'*55}")

    expected = {
        "classify": ("easy", "0.75 – 0.90"),
        "prioritize": ("medium", "0.45 – 0.65"),
        "reply": ("hard", "0.25 – 0.50"),
    }

    for r in results:
        diff, exp_range = expected.get(r["task_id"], ("?", "?"))
        print(
            f"  {r['task_id']:<12} {diff:<12} {r['steps']:<8} {r['episode_score']:<10.4f} {exp_range}"
        )

    print(f"\n  Overall mean: {sum(r['episode_score'] for r in results)/len(results):.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
