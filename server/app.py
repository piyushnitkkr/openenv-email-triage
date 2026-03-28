"""
FastAPI application exposing the Email Triage environment as an HTTP API.

Endpoints:
  GET  /              → health check
  GET  /tasks         → list all tasks with action schemas
  POST /reset         → start new episode
  POST /step          → execute one action
  GET  /state         → current environment state
  POST /grader        → grade a full episode history
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path so imports work both locally and in Docker
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import EmailAction, EmailObservation, EmailReward, EmailState
from server.email_environment import EmailEnvironment
from server.tasks import TASK_REGISTRY

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenEnv — Email Triage",
    description=(
        "An OpenEnv-compatible environment for evaluating LLM agents on realistic "
        "email triage tasks: classification, inbox prioritization, and reply drafting."
    ),
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (stateful per session)
env = EmailEnvironment()

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepResponse(BaseModel):
    observation: EmailObservation
    reward: EmailReward
    done: bool
    info: dict[str, Any] = {}


class GraderRequest(BaseModel):
    """Run a full grader over episode history (for offline scoring)."""
    task_id: str
    history: list[dict[str, Any]]


class GraderResponse(BaseModel):
    task_id: str
    episode_score: float
    step_scores: list[float]
    feedback: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", summary="Health check")
def health_check() -> dict[str, str]:
    return {"status": "ok", "environment": "email-triage"}


@app.get("/tasks", summary="List all available tasks with schemas")
def list_tasks() -> dict[str, Any]:
    tasks = []
    for task_id, task in TASK_REGISTRY.items():
        tasks.append({
            "id": task.task_id,
            "difficulty": task.difficulty,
            "description": task.description,
            "max_steps": task.max_steps,
            "required_action_type": task.required_action_type,
            "action_description": task.action_description,
            "action_schema": {
                "action_type": task.required_action_type,
                "label": "spam | urgent | normal | newsletter  (classify only)",
                "priority_order": "list[email_id]  (prioritize only)",
                "reply_text": "string  (reply only)",
                "reasoning": "string  (optional)",
            },
        })
    return {"tasks": tasks, "total": len(tasks)}


@app.post("/reset", response_model=EmailObservation, summary="Start a new episode")
def reset_episode(request: ResetRequest) -> EmailObservation:
    try:
        obs = env.reset(task_id=request.task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse, summary="Execute one action")
def step_episode(action: EmailAction) -> StepResponse:
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state", summary="Get current environment state")
def get_state() -> dict[str, Any]:
    return env.state


class BaselineResponse(BaseModel):
    scores: list[dict[str, Any]]
    mean_score: float


@app.get("/baseline", response_model=BaselineResponse, summary="Trigger baseline inference script")
def run_baseline_script() -> BaselineResponse:
    import subprocess
    import json
    import os

    hf_token = os.environ.get("HF_TOKEN", "")
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o")

    if not hf_token:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set in environment")

    try:
        # Run inference.py (root-level script, required by submission spec)
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=1200,  # 20 min limit per spec
            env={
                **os.environ,
                "HF_TOKEN": hf_token,
                "API_BASE_URL": api_base_url,
                "MODEL_NAME": model_name,
                "ENV_BASE_URL": "http://localhost:7860",
            }
        )

        # Read the dumped JSON scores
        if os.path.exists("baseline_results.json"):
            with open("baseline_results.json", "r") as f:
                data = json.load(f)
            return BaselineResponse(scores=data["scores"], mean_score=data["mean_score"])
        else:
            raise HTTPException(
                status_code=500,
                detail=f"inference.py failed or produced no output.\nStdout: {result.stdout[-500:]}\nStderr: {result.stderr[-500:]}"
            )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Inference script timed out (20 min limit)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/grader", response_model=GraderResponse, summary="Grade an episode from history")
def grade_episode(request: GraderRequest) -> GraderResponse:
    """
    Given an episode history (list of {action_type, label, priority_order, reply_text, score}),
    compute the final graded score.
    """
    from server.graders import BILLING_DISPUTE_RUBRIC, grade_classification, grade_prioritization, grade_reply
    from server.tasks import TASK_REGISTRY

    if request.task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{request.task_id}'")

    step_scores: list[float] = []
    feedbacks: list[str] = []

    for entry in request.history:
        score = entry.get("score", 0.0)
        step_scores.append(float(score))
        feedbacks.append(entry.get("feedback", ""))

    episode_score = sum(step_scores) / len(step_scores) if step_scores else 0.0

    return GraderResponse(
        task_id=request.task_id,
        episode_score=round(episode_score, 4),
        step_scores=step_scores,
        feedback=" | ".join(feedbacks[:3]),
    )


# ---------------------------------------------------------------------------
# Entry point (required by openenv validate)
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the 'server' script defined in pyproject.toml."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
