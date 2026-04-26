---
title: Enterprise AP Environment
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# Enterprise AP Environment

**An OpenEnv-compatible RL/Agent training environment**
**Team SHIPWithTEA : Prachi and Dharmendra**

The AI agent is dropped into a realistic enterprise **Accounts Payable department** and must process invoices by interacting across multiple applications: an email inbox, an ERP system (with schema drift), and vendor communication channels.

---

## Key Features

- **Multi-App Workflow:** Forces the agent to synthesize information across different systems (unstructured email inbox and structured ERP database).
- **Adversarial Security (Phishing):** Tests the agent's attention to detail by deploying lookalike-domain phishing attacks (e.g., `techsuppIies.com` vs. `techsupplies.com`).
- **Dynamic Schema Drift:** The ERP API silently alters its required fields mid-episode, requiring the agent to parse error messages, recover, and adapt dynamically.
- **Multi-Turn Interactive Resolution:** Agents don't just extract data; they identify discrepancies and email a simulated reactive vendor actor to solicit a corrected invoice, then re-process the updated data.
- **TRL/Unsloth Training Integration:** Proves learnability with an integrated Hugging Face TRL & Unsloth `GRPOTrainer` pipeline.

---

## Quick Links
- **[HuggingFace Writeup & Details](https://huggingface.co/Prachi-2601/Multi-App-RL-Env-Invoice-Processing-Schema-Drift-Fraud-Detection-Vendor-Negotiation)**
- **[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/decent-cow26/invoice-env)**

---

##  The Problem
Enterprise Accounts Payable (AP) departments process hundreds of invoices every day. The work looks repetitive on the surface — read the invoice, check it against the Purchase Order, approve or reject — but the reality is far more complicated. Vendors send incorrect amounts. ERP systems change their APIs without warning. Duplicate invoices slip through undetected. And increasingly, attackers send fraudulent invoices from lookalike domains that differ from the real vendor by a single character.

Current rule-based automation systems catch obvious errors but fail at anything that requires judgment. We built this environment to see if a language model could be trained to handle this entire workflow — and get measurably better at it over time.

---

##  Why It Matters
Most RL environments for language models test a single capability — math reasoning, code generation, or factual retrieval. This environment tests a compound skill: multi-step, multi-app reasoning under adversarial conditions. The failure modes (approving a fraudulent invoice, missing a duplicate, failing to negotiate) are the exact failures that cost real organizations real money.

---

## Hackathon Themes Covered

| Theme | Coverage |
|-------|----------|
| **#3.1 World Modeling – Professional Tasks** | Full multi-app enterprise AP workflow |
| **Scaler AI Labs – Multi-App RL** | Agent bridges unstructured email + structured ERP API |
| **Patronus AI – Schema Drift** | ERP API silently changes its required fields mid-workflow |
| **#1 Multi-Turn Environment Dynamics** | Agent resolves a price discrepancy via multi-turn email negotiation with a simulated reactive vendor actor |
| **Safety / Security** | Phishing/lookalike domain fraud detection |

---

## Tasks (Progressive Difficulty)

| Task | Difficulty | Description |
|------|-----------|-------------|
| `easy` | Beginner | Read email → Query ERP → Extract 8 fields → Approve |
| `medium` | Medium | Same as easy, but detect a subtle line-item price mismatch |
| `hard` | Hard | **Schema Drift**: ERP rejects `vendor_name`, requires `vendor_tax_id`. Also detect duplicate invoice. |
| `expert_negotiation` | Expert | Invoice is overpriced. **Email the vendor** (simulated reactive actor), get a corrected invoice dynamically injected into the inbox, re-extract all fields, then approve. |
| `expert_fraud` | Expert | Sender is `@techsuppIies.com` (lookalike) AND uses a fraudulent IBAN. Flag both and reject. |

---

## Action Space

| Action | Fields | Description |
|--------|--------|-------------|
| `read_email` | `email_id` | Open an email from the inbox |
| `query_erp` | `api_endpoint`, `api_payload` | Query the ERP database |
| `extract` | `field_name`, `field_value` | Store an extracted invoice field |
| `match_po` | — | Cross-reference invoice total vs PO |
| `flag` | `field_name` | Raise an issue flag (`price_mismatch`, `fraud`, `fraud_iban`, `duplicate_invoice`, `tax_mismatch`) |
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
- Negotiation email sent to vendor (triggers dynamic inbox update): **+0.20**
- Wrong actions: **-0.01 to -0.05**

### Final Grading (40 / 30 / 30)
| Component | Weight |
|-----------|--------|
| Field extraction accuracy (8 fields) | 40% |
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

# Run the Gradio demo UI (in-process, no server needed)
python app/demo.py

# Run adaptive curriculum training (rule-based reference agent)
python train.py --episodes 60

# Run real LLM inference agent (requires HF_TOKEN)
export HF_TOKEN="your_token_here"          # PowerShell: $env:HF_TOKEN="..."
python inference.py --all --episodes 10    # all 5 tasks
python inference.py --task expert_fraud    # single task

# Plot results from any run
python plot_llm_results.py                 # auto-detects latest *_results.json
python plot_llm_results.py 8B_results.json # or specify a file
```

## Docker (for HuggingFace Spaces)

```bash
docker build -t enterprise-ap-env .
docker run -p 7860:7860 enterprise-ap-env
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset?task_name=easy` | POST | Start a new episode — returns a unique `session_id` |
| `/step?session_id=<uuid>` | POST | Take one action — pass `session_id` from `/reset` |
| `/state?session_id=<uuid>` | GET | Inspect live state of an active session |
| `/sessions` | GET | List all active sessions and their task names |
| `/tasks` | GET | List all available tasks |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

---

## Try It Interactively

Once the server is running (see Quick Start above), you can drive the environment with plain HTTP calls. The full Swagger UI is available at **`http://localhost:7860/docs`**.

Here is a complete `expert_fraud` episode in three curl commands:

```bash
# 1. Start a new episode
curl -s -X POST "http://localhost:7860/reset?task_name=expert_fraud" | python -m json.tool

# 2. Read the email (note the sender — is it a real domain?)
curl -s -X POST "http://localhost:7860/step?session_id=<YOUR_SESSION_ID>" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "read_email", "email_id": "email_006"}' | python -m json.tool

# 3. Query ERP and compare IBANs — then flag and reject
curl -s -X POST "http://localhost:7860/step?session_id=<YOUR_SESSION_ID>" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "query_erp", "api_endpoint": "/api/v1/po", "api_payload": {"vendor_name": "TechSupplies Inc."}}' | python -m json.tool
```

> Replace `<YOUR_SESSION_ID>` with the `session_id` returned by the `/reset` call.

---

## Project Structure

```
enterprise-ap-env/
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

## Evaluation Results

### Reference Agent Validation — Discriminative Power (100 Episodes)

A deterministic rule-based agent run at two noise levels validates the environment's discriminative power.
All tasks exceed their target thresholds, confirming reward shaping works as designed.

| Task | Weak Agent | Strong Agent | Gap | Target | Status |
|------|-----------|-------------|-----|--------|--------|
| easy | 0.82 | 0.94 | +12% | 0.85 | PASS |
| medium | 0.83 | 0.95 | +12% | 0.75 | PASS |
| hard | 0.77 | 0.95 | +18% | 0.65 | PASS |
| expert_negotiation | 0.79 | 0.96 | +17% | 0.70 | PASS |
| expert_fraud | 0.65 | 0.86 | +21% | 0.70 | PASS |
| **Average** | **0.77** | **0.93** | **+16%** | — | — |

![Reward Curves — Strong Agent (low noise)](./70B_curves.png)
![Reward Curves — Weak Agent (high noise)](./8B_curves.png)

> The environment reliably separates weak and strong agents. The hardest tasks show the largest gaps —
> exactly what a well-designed benchmark should do.

---

### LLM Agent (`inference.py`) — Real Benchmark Results

A real LLM agent (`meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Inference API)
drives the environment via the OpenAI-compatible REST interface with no task-specific hardcoding.

```bash
# Run a real LLM benchmark (requires HF_TOKEN env var)
export HF_TOKEN="your_token_here"
python inference.py --all --episodes 10
python plot_llm_results.py   # auto-detects the output file
```

| Task | LLM Score | Steps | Result |
|------|-----------|-------|--------|
| easy | **0.99** | 14 | PASS — extracted all 8 fields, approved correctly |
| expert_fraud | **0.99** | 14 | PASS — detected lookalike domain, flagged `fraud` + `fraud_iban`, rejected |

Key observations from the `expert_fraud` run:
- LLM read email from `billing@vertx.com` (lookalike for `vertex.com`)
- Extracted all 8 invoice fields correctly (including the attacker's fraudulent IBAN)
- Independently flagged `fraud` and `fraud_iban` — no hints given
- Issued `reject` decision — episode complete, score 0.99

> **The environment is the research contribution.** The reference agent validates reward correctness.
> The LLM agent establishes the actual baseline — 0.99 on clean and fraud detection tasks.

Built for the **Meta AI Hackathon Grand Finale 2026**.
