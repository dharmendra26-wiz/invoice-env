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
- **GRPO RL Training Pipeline:** `colab_rl_training.ipynb` trains `llama-3-8b-Instruct` (4-bit, Unsloth) using TRL's `GRPOTrainer`. The reward function is distilled directly from `environment.py` — training connects to the same reward logic as evaluation.

---

## Quick Links
- **[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/decent-cow26/invoice-env)**
- **[HuggingFace Writeup & Model Card](https://huggingface.co/Prachi-2601/Multi-App-RL-Env-Invoice-Processing-Schema-Drift-Fraud-Detection-Vendor-Negotiation)**
- **[GRPO Training Notebook (Colab)](./colab_rl_training.ipynb)** — Trains `llama-3-8b-Instruct` via Unsloth + TRL GRPOTrainer

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

## Gradio Demo — Two Agent Modes

The Gradio dashboard (`python app/demo.py`) supports two modes and **switches automatically** based on whether you provide an HF token:

| Mode | How to activate | Who plays the episode |
|------|----------------|----------------------|
| **Scripted Agent** (default) | Open the Space or run `python app/demo.py` — no token needed | A deterministic rule-based reference agent plays automatically. Always works. |
| **Autonomous LLM Agent** | Paste your HuggingFace token + model name in the sidebar and click Run | Any HF-compatible LLM drives every action live — Llama, Mistral, Qwen, your own fine-tuned model, anything. |

> No token → scripted agent runs automatically. Token + model name → your chosen LLM takes over. The environment, reward function, and observation format are **identical in both modes** — plug in any model and benchmark it instantly, no integration code needed.

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

## RL Training Pipeline (Unsloth + TRL GRPO)

The Colab notebook [`colab_rl_training.ipynb`](./colab_rl_training.ipynb) trains `llama-3-8b-Instruct` (4-bit quantized) to solve the AP environment using **GRPO** (Group Relative Policy Optimization).

**How it works:**
1. For each AP workflow state prompt, the model generates **4 candidate JSON actions**.
2. Each action is scored by the AP environment reward function (distilled from `environment.py`).
3. GRPO updates the model to assign higher probability to higher-reward actions — reward trending up, loss trending down.

| Component | Choice |
|-----------|--------|
| Base model | `unsloth/llama-3-8b-Instruct-bnb-4bit` (4-bit, ~5 GB) |
| Fine-tuning | LoRA r=16 (~1% trainable params) |
| RL algorithm | GRPOTrainer (TRL) — no reference model needed |
| Dataset | 125 synthetic AP workflow state prompts |
| GPU requirement | Free T4 (~20–35 min for 100 steps) |

**Training progress — Loss ↓ and Reward ↑ over 100 steps:**

![GRPO Training Curves — Llama-3.1-8B + LoRA on Colab T4](./grpo_training_curves.png)

> Training loss drops from ~0.0025 → **0.001** final. Average reward climbs from near 0.2 and stabilises at **0.48–0.65**, approaching the 0.70 target threshold. Both trends confirm the AP environment produces a genuine, learnable gradient signal.

**Before vs. After GRPO fine-tuning (peak episode reward):**

![RL Results — Before vs After Training](./rl_training_results_final.png)

| Task | Before Training | After GRPO | Improvement |
|------|----------------|------------|-------------|
| Easy | 0.26 | **0.88** | +0.62 (+238%) |
| Expert Negotiation | 0.18 | **0.78** | +0.60 (+333%) |

**Run it yourself:** Open `colab_rl_training.ipynb` in Google Colab → Runtime → T4 GPU → Run All.

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

**25 real episodes, 5 per task — `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Router API:**

| Task | Avg Score | Result | Key Behaviour |
|------|-----------|--------|---------------|
| easy | **0.10** | FAIL | Correctly extracts all 8 fields + queries ERP, but then hallucinates spurious flags instead of approving — never issues `approve` before hitting the 25-step limit |
| medium | **0.99** | PASS | Price mismatch detected and rejected in all 5 episodes, 13–14 steps |
| hard | **0.99** | PASS | Schema drift handled perfectly in all 5 episodes — v1 ERP rejected, auto-retries with `vendor_tax_id` on v2 |
| expert_negotiation | **0.23** | FAIL | Model sends `send_email` to vendor (picks up the intent) but cannot close the multi-turn negotiation loop before step limit |
| expert_fraud | **0.88** | PASS | IBAN fraud flagged and rejected in all 5 episodes; 1/5 episodes also caught the lookalike domain (0.99) |
| **Average** | **0.64** | — | — |

![Real LLM Reward Curves — Llama-3.1-8B-Instruct, 5 episodes per task](./Llama_3.1_8B_Instruct_real_curves.png)

The benchmark produces a **capability profile**, not a single score. The 8B model excels at pattern-detection and schema-recovery tasks (Hard: 0.99, Medium: 0.99, Fraud: 0.88) but fails at workflow adherence (Easy: can't `approve`) and multi-turn planning (Negotiation: can't close the loop). This is exactly the discriminative signal a useful benchmark should provide.

> **The environment is the research contribution.** It exposes precisely where a model breaks down under enterprise conditions — and which failure modes GRPO fine-tuning should target.

Built for the **Meta AI Hackathon Grand Finale 2026**.
