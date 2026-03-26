---
title: OpenEnv Email Triage
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
tags:
  - openenv
---
# OpenEnv Email Triage

An **OpenEnv-compatible** environment where an LLM agent reads, categorizes, prioritizes, and responds to emails in a realistic inbox. Built for the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework by Meta-PyTorch × Hugging Face.

---

## Motivation

Email triage is something humans do every day — it's immediately understandable, measurable, and reflects real cognitive work. Unlike toy gridworlds or puzzles, this environment evaluates agents on:
- **Semantic understanding** (is this spam or urgent?)
- **Multi-criteria reasoning** (how do I rank 10 emails by urgency?)
- **Constrained generation** (draft a professional reply that addresses the right facts)

It fills a gap in OpenEnv: realistic NLP-heavy workloads with deterministic graders.

---

## Environment Description

Each episode consists of the agent interacting with a simulated inbox. The agent receives observations (email content) and must submit structured actions (classify, prioritize, reply). The environment grades each action against ground truth and returns shaped rewards.

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Active task: `classify`, `prioritize`, or `reply` |
| `step_number` | int | Current step in episode |
| `email_id` | string | ID of current email |
| `subject` | string | Email subject line |
| `sender` | string | Sender email address |
| `timestamp` | string | ISO 8601 timestamp |
| `body` | string | Full email body |
| `thread_history` | list[str] | Prior messages in thread (oldest first) |
| `inbox_context` | list[InboxEmail] | All emails summarized (prioritize task only) |
| `task_description` | string | Human-readable task instructions |
| `is_done` | bool | Whether episode is complete |

---

## Action Space

| Field | Type | Required for |
|-------|------|-------------|
| `action_type` | `classify` \| `prioritize` \| `reply` | Always |
| `label` | `spam` \| `urgent` \| `normal` \| `newsletter` | `classify` |
| `priority_order` | list[email_id] | `prioritize` |
| `reply_text` | string | `reply` |
| `reasoning` | string | Optional (not scored) |

---

## Tasks

### Task 1 — Email Classification (Easy)
- **Input:** 10 emails presented one at a time
- **Action:** `classify` with one of 4 labels
- **Grader:** Exact match = 1.0, near-miss (e.g., urgent↔normal) = 0.5, wrong = 0.0
- **Expected GPT-4o score:** 0.75 – 0.90

### Task 2 — Inbox Prioritization (Medium)
- **Input:** 10 emails shown all at once in inbox
- **Action:** `prioritize` with ranked list of email IDs
- **Grader:** Normalized Kendall tau distance, top-3 positions weighted 2×
- **Expected GPT-4o score:** 0.45 – 0.65

### Task 3 — Reply Drafting (Hard)
- **Input:** Multi-turn billing dispute thread
- **Action:** `reply` with draft response
- **Grader:** Rubric keyword matching — acknowledgment, invoice reference, refund commitment, timeline, professional close, no hallucinated facts
- **Expected GPT-4o score:** 0.25 – 0.50

---

## Reward Function

Rewards are shaped per step in [0.0, 1.0]:
- **Exact label match:** 1.0
- **Near-miss classification:** 0.5 (e.g., urgent vs. normal)
- **Wrong action type:** 0.0 + info message
- **Repeated identical action:** 0.0 (penalty)
- **Exceed max steps:** episode ends, final score averaged
- **Episode score:** mean of all step scores

---

## Setup Instructions

### Local Python Setup

```bash
cd openenv-email-triage

# Install dependencies
pip install fastapi uvicorn pydantic

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Test it
curl http://localhost:8000/
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id": "classify"}'
```

### Run Tests

```bash
pytest tests/ -v
```

### Run Baseline Script

```bash
export OPENAI_API_KEY=sk-...
python baseline/run_baseline.py --model gpt-4o --seed 42
```

### Docker Setup

```bash
docker build -t openenv-email-triage .
docker run -p 8000:8000 openenv-email-triage
curl http://localhost:8000/
```

---

## Baseline Scores (Reference)

Run with GPT-4o, seed=42, 2024-01-15:

| Task | Difficulty | Expected Score |
|------|------------|----------------|
| Classification | Easy | 0.75 – 0.90 |
| Prioritization | Medium | 0.45 – 0.65 |
| Reply Drafting | Hard | 0.25 – 0.50 |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/docs` | Swagger UI |
| GET | `/tasks` | List all tasks |
| POST | `/reset` | Start episode (body: `{"task_id": "classify"}`) |
| POST | `/step` | Submit action |
| GET | `/state` | Current state snapshot |
| POST | `/grader` | Grade episode from history |

---

## HF Space URL

> _To be filled after deployment_: `https://huggingface.co/spaces/YOUR_USERNAME/openenv-email-triage`

---

## `openenv.yaml` Summary

```yaml
name: email-triage
version: 1.0.0
tasks: [classify (easy), prioritize (medium), reply (hard)]
reward_range: [0.0, 1.0]
observation_space: email content + inbox context
action_space: classify | prioritize | reply
```
