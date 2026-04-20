---
title: Invoice Processing Environment
emoji: 🧾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---
# Invoice Processing Environment

An OpenEnv-compatible AI agent environment for invoice processing and accounts payable validation.

## Description

This environment simulates a real-world accounts payable workflow where an AI agent must:
- Extract structured fields from invoice text
- Match invoices against purchase orders
- Detect discrepancies (price mismatches, duplicate invoices, tax errors)
- Make approve/reject decisions

## Action Space

| Action Type | Fields | Description |
|----|----|----|
| `extract` | `field_name`, `field_value` | Extract a field from the invoice |
| `match_po` | none | Match invoice total against PO |
| `flag` | `field_name` | Flag an issue (price_mismatch, duplicate_invoice, tax_mismatch) |
| `match_duplicate` | none | Check if invoice was previously processed |
| `approve` | none | Approve the invoice |
| `reject` | none | Reject the invoice |

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
| `easy` | Easy | Extract fields from clean invoice, match PO, approve |
| `medium` | Medium | Detect price mismatch between invoice and PO |
| `hard` | Hard | Detect duplicate invoice + tax miscalculation |

## Reward Function

- Correct field extraction: +0.07 per field
- Wrong field value: -0.02
- Correct flag raised: +0.12
- Wrong flag: -0.05
- Successful PO match: +0.10
- Final grader score: up to 1.0 (based on fields + flags + decision)

## Baseline Scores

| Task | Score |
|---|---|
| easy | ~0.85 |
| medium | ~0.70 |
| hard | ~0.55 |

## Setup
```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Run Inference
```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
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
 
