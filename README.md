---
title: Enterprise Multi-App AI Simulator
emoji: 🏢
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---
# Enterprise Multi-App AI Simulator 🏢

A cutting-edge, OpenEnv-compatible RL/Agent training environment built specifically for the **Meta Hackathon Grand Finale**. This environment models complex, messy, real-world enterprise workflows.

Instead of parsing static text, the AI agent is dropped into an enterprise system where it must interact across multiple applications, handle unexpected system changes, and communicate with external agents.

## 🏆 Hackathon Themes Targeted
1. **Theme #3.1 (World Modeling - Professional Tasks)**: Simulates a multi-application enterprise environment (Email Inbox + ERP Database).
2. **Scaler AI Labs Bonus (Multi-App Workflows)**: The agent must bridge data between an unstructured Email inbox and a structured ERP API.
3. **Patronus AI Bonus (Schema Drift)**: The ERP API changes its schema mid-workflow, forcing the agent to adapt.
4. **Theme #1 (Multi-Agent Interactions)**: The agent must negotiate with a simulated vendor via email to resolve discrepancies.
5. **Theme #2 (Safety & Security)**: Tests the agent's ability to detect sophisticated phishing/lookalike domain fraud.

## Action Space

The agent has a rich, multi-app action space:

| Action Type | Fields | Description |
|---|---|---|
| `read_email` | `email_id` | Open an email from the inbox |
| `query_erp` | `api_endpoint`, `api_payload` | Query the company ERP database |
| `extract` | `field_name`, `field_value` | Extract data to internal memory |
| `match_po` | none | Cross-reference invoice with ERP PO |
| `send_email` | `email_id`, `email_subject`, `email_body` | Email a vendor to negotiate / ask for corrections |
| `flag` | `field_name` | Flag an issue (fraud, price_mismatch, duplicate_invoice) |
| `approve` / `reject` | none | Final workflow decision |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `invoice_text` | string | Raw invoice text |
| `po_data` | object | Purchase order reference data |
| `extracted_fields` | object | Fields extracted so far |
| `flags` | array | Issues flagged so far |
| `current_step` | integer | Current step number |
| `message` | string | Feedback from last action |

## Tasks

| Task | Difficulty | Description |
|---|---|---|
| `easy` | Easy | Read email, query ERP, extract fields, approve. |
| `medium` | Medium | Read email, query ERP, detect a subtle price mismatch. |
| `hard` | Hard | **Schema Drift**. ERP rejects standard query and requires a new parameter (`tax_id`). |
| `expert_negotiation`| Expert | **Multi-Agent**. Agent finds a mismatch, uses `send_email` to negotiate with vendor, environment dynamically generates a response email with corrected invoice. |
| `expert_fraud` | Expert | **Security**. Perfect invoice, but sender email is a phishing lookalike (`@techsuppIies.com`). Agent must flag as fraud. |

## Setup & Running
```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Docker
```bash
docker build -t invoice-env .
docker run -p 7860:7860 invoice-env
```

## API Endpoints

- `POST /reset?task_name=easy` — Start new episode
- `POST /step?task_name=easy` — Take an action
- `GET /state?task_name=easy` — Get current state
- `GET /tasks` — List all tasks
- `GET /health` — Health check
- `GET /docs` — Interactive API documentation
