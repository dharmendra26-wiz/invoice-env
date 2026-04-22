---
title: Enterprise Multi-App AI Simulator
emoji: 🏢
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# Enterprise Multi-App AI Simulator 🏢

An OpenEnv-compatible RL/Agent training environment for the **Meta AI Hackathon Grand Finale**.

The AI agent is dropped into a realistic enterprise **Accounts Payable department** and must process invoices by interacting across multiple applications: an email inbox, an ERP system (with schema drift), and vendor communication channels.

---

## Hackathon Themes Covered

| Theme | Coverage |
|-------|----------|
| **#3.1 World Modeling – Professional Tasks** | Full multi-app enterprise AP workflow |
| **Scaler AI Labs – Multi-App RL** | Agent bridges unstructured email + structured ERP API |
| **Patronus AI – Schema Drift** | ERP API silently changes its required fields mid-workflow |
| **#1 Multi-Agent Interactions** | Agent negotiates with a simulated vendor via email |
| **Safety / Security** | Phishing/lookalike domain fraud detection |

---

## Tasks (Progressive Difficulty)

| Task | Difficulty | Description |
|------|-----------|-------------|
| `easy` | Beginner | Read email → Query ERP → Extract 7 fields → Approve |
| `medium` | Medium | Same as easy, but detect a subtle line-item price mismatch |
| `hard` | Hard | **Schema Drift**: ERP rejects `vendor_name`, requires `vendor_tax_id`. Also detect duplicate invoice. |
| `expert_negotiation` | Expert | Invoice is overpriced. **Email the vendor**, get a corrected invoice, re-extract, approve. |
| `expert_fraud` | Expert | Perfect invoice but sender is `@techsuppIies.com` (capital-I lookalike). Flag as fraud. |

---

## Action Space

| Action | Fields | Description |
|--------|--------|-------------|
| `read_email` | `email_id` | Open an email from the inbox |
| `query_erp` | `api_endpoint`, `api_payload` | Query the ERP database |
| `extract` | `field_name`, `field_value` | Store an extracted invoice field |
| `match_po` | — | Cross-reference invoice total vs PO |
| `flag` | `field_name` | Raise an issue flag (`price_mismatch`, `fraud`, `duplicate_invoice`, `tax_mismatch`) |
| `match_duplicate` | — | Check if invoice was previously processed |
| `send_email` | `email_id`, `email_subject`, `email_body` | Email a vendor to negotiate |
| `approve` / `reject` | — | Final decision — ends the episode |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `inbox_status` | array | List of unread emails (id, sender, subject) |
| `email_content` | string | Full body of currently opened email |
| `erp_response` | object | Response from last ERP query |
| `extracted_fields` | object | Fields extracted so far |
| `flags` | array | Issues flagged so far |
| `current_step` | integer | Current step number |
| `message` | string | Feedback from the last action |

---

## Reward System

### Step-Level Rewards
- Read email: **+0.05**
- ERP query success: **+0.10**
- Correct field extraction: **+0.07**
- Correct flag: **+0.12**
- Negotiation email (triggers reply): **+0.20**
- Wrong actions: **-0.01 to -0.05**

### Final Grading (40 / 30 / 30)
| Component | Weight |
|-----------|--------|
| Field extraction accuracy (7 fields) | 40% |
| Correct flags raised / negotiation done | 30% |
| Correct approve/reject decision | 30% |

> If ERP was never queried: hard floor of **0.10** (enforces multi-app workflow).

---

## Data Randomization

Every `reset()` generates a **fresh, unique episode**:
- 5-vendor pool (TechSupplies, OfficeMart, GlobalTech, Vertex Software, CloudServe)
- Randomized line items, quantities, prices, tax rates (8–15%), dates (2024–2025)
- Random invoice numbers
- No two episodes are identical — LLMs cannot memorize answers

---

## Quick Start

```bash
pip install -r requirements.txt

# Start the REST API server
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860

# Run training (local mode, no server needed)
python train.py --episodes 60

# Launch Gradio live demo
python app/demo.py

# Run LLM inference agent
python inference.py --all
```

## Docker (for HuggingFace Spaces)

```bash
docker build -t invoice-env .
docker run -p 7860:7860 invoice-env
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset?task_name=easy` | POST | Start a new episode |
| `/step?task_name=easy` | POST | Take one action |
| `/state?task_name=easy` | GET | Inspect current state |
| `/tasks` | GET | List all tasks |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

---

## Project Structure

```
invoice-env/
├── app/
│   ├── main.py          # FastAPI server
│   ├── environment.py   # Core state machine
│   ├── tasks.py         # Randomized task generator + grader
│   ├── models.py        # Pydantic data models
│   └── demo.py          # Gradio live demo UI
├── train.py             # Training script + reward curves
├── inference.py         # LLM agent (HuggingFace)
├── openenv.yaml         # OpenEnv specification
├── Dockerfile           # For HuggingFace Spaces
└── requirements.txt
```

---

## Training Results

All 5 tasks achieve **0.94 final average reward** with clear upward learning curves.

Built for the **Meta AI Hackathon Grand Finale 2026**.
