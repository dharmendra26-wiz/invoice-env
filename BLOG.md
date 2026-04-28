# Enterprise AP-Env: Turning fraud into foresight

*Meta AI Hackathon Grand Finale 2026 · Team SHIPWithTEA — Prachi & Dharmendra*


 Problem: Business Email Compromise layered on top of duplicate invoicing layered on top of lookalike domain fraud — costs enterprises **$2.9 billion every year**. And yet, until now, there has been no standardised environment to test whether an AI agent would have caught it.

We built that environment. Then we trained an LLM inside it and watched it learn.

---

> **For judges: nothing in this demo is scripted or static.**
> Every episode generates a new vendor, new invoice, new prices, new fraud IBAN (using OS-level entropy via Python's `secrets` module) — the agent cannot memorise answers, it must reason live. When you open the Gradio demo, you can plug in **any HuggingFace-compatible model** (Llama, Mistral, Qwen, your own fine-tuned checkpoint) and watch it play a real episode autonomously. If no token is provided, a clearly-labelled **Rule-Based Reference Agent** runs instead so you can observe the environment without an API key — the UI badge makes this explicit.

---

## Try It Right Now

| Resource | Link |
|----------|------|
| **Live Demo (HuggingFace Space)** | [spaces/decent-cow26/invoice-env](https://huggingface.co/spaces/decent-cow26/invoice-env) |
| (Multi-App-RL-Env-Invoice-Processing-Schema-Drift-Fraud-Detection-Vendor-Negotiation) |
| **GRPO Training Notebook (Colab T4)** | [`colab_rl_training.ipynb`](./colab_rl_training.ipynb) — full RL loop in ~35 min |
| **GitHub Source** | *(https://github.com/dharmendra26-wiz/Enterprise-AP-Environment)* |

---

## The Problem We Are Solving

The AI industry is racing to deploy LLMs into enterprise Accounts Payable departments. Automate the invoice. Cut headcount. Process payments faster.

But there is a question nobody is asking before they flip the switch: **what happens when the invoice is fake?**

MMLU tests trivia. SWE-bench tests code. Neither tests whether an LLM will rubber-stamp a $84,000 payment to a fraudster. There is no standard environment, no shared benchmark, no reproducible test harness for enterprise financial agent safety.

**Enterprise AP-Env is that benchmark.** A procedurally generated, OpenEnv-compliant, multi-application simulation that forces AI agents through the exact failure modes that cost enterprises $2.9 billion a year.

---

## Full Feature List

### Backend & API
- **UUID-based session isolation** — every `/reset` call generates a fresh `uuid4` session ID; no two agents ever share state
- **300-second TTL eviction** — idle sessions are automatically garbage-collected via monotonic clock, preventing memory leaks from abandoned episodes
- **Thread-safe parallel sessions** — multiple LLMs or users can run simultaneous episodes against the same server with zero state collision
- **FastAPI + Uvicorn** — production-grade ASGI backend with auto-generated OpenAPI docs at `/docs`
- **OpenAI-compatible REST interface** — any LLM client that speaks `/v1/chat/completions` can drive the environment without custom integration

### Environment & Task Engine
- **OpenEnv-compliant state machine** — inherits `openenv-core` base class, exposing standard `reset()` / `step()` / `state` Gym-style API
- **5-vendor procedural pool** — vendor names, domains, fraud domains, IBANs, and item catalogues rotate across 5 distinct companies every episode
- **Randomised line items** — item quantities, unit prices, subtotals, tax rates (8%, 10%, 12%, 15%), and invoice dates are all independently randomised
- **Cryptographic fraud IBANs** — fraudulent bank accounts generated with Python's `secrets` module (OS-level entropy), impossible to pre-compute or memorise
- **Capital-I lookalike domains** — fraud domains use visually identical Unicode substitutions: `techsuppIies.com`, `0fficemart.ltd`, `g1obaltech.com`, `vertx.com`, `c1oudserve.io`
- **Hard ERP floor in reward** — if the agent never queries the ERP, max possible score is capped at **0.10** regardless of email extraction quality, enforced in `grade_task()`
- **30-step hard termination** — fixed step budget per episode; environment force-closes at step 30 to prevent infinite loops
- **Schema drift injection** — on the `hard` task, `vendor_name` is silently deprecated mid-session; agent must read the API error and switch to `vendor_tax_id`
- **Simulated vendor email injection** — on `expert_negotiation`, a corrected invoice is injected directly into the inbox after the agent sends a negotiation email
- **Duplicate invoice detection** — the `hard` task embeds a duplicate invoice number that must be flagged before approval
- **Multi-turn negotiation state tracking** — `self.negotiated` flag tracked separately so the reward function grades negotiation as a distinct behaviour

### Reward System
- **Granular step-level shaping** — `+0.05` read email, `+0.10` ERP query, `+0.07` per correct field extracted, `+0.12` flag match/duplicate, `+0.20` send negotiation email
- **3-axis final rubric** — Extraction Accuracy 40%, Workflow Flags 30%, Final Decision 30%
- **Non-binary scoring** — no 0/1 reward; every correct action earns a signal, making the gradient informative for RL training
- **Decision penalty** — approving a fraudulent invoice or rejecting a legitimate one scores 0.0 on the decision axis (30% of the episode score)

### Training & Curriculum
- **Adaptive curriculum in `train.py`** — 5-episode rolling average auto-promotes agent to harder tasks at >88% and demotes at <50%
- **Direct env mode (no HTTP)** — `train.py` runs the environment in-process without a server, cutting episode latency to near zero
- **HTTP mode for LLM benchmarking** — `inference.py` drives the live FastAPI server over HTTP using any HuggingFace Router model
- **Sliding context window** — `inference.py` keeps only the last 16 messages (`messages[:2] + messages[-16:]`) to prevent LLM context overflow on long episodes
- **Exponential backoff retry** — LLM API calls retry up to 4 times with doubling wait (15s → 30s → 60s → 120s) on 429/500/502/503/504 errors
- **GRPO RL training on Colab T4** — full reinforcement learning loop runs in ~35 minutes on a free T4 GPU using Unsloth 4-bit + TRL GRPOTrainer
- **LoRA fine-tuning** — rank-16 LoRA adapters update only 1–2% of model parameters, keeping peak GPU memory under 12 GB

### Demo & UI
- **Dual-mode Gradio UI** — no token = Rule-Based Reference Agent (green badge); token provided = real LLM Agent with your model name (blue badge)
- **Any HuggingFace model supported** — plug in Llama, Mistral, Qwen, Falcon, or your own fine-tuned checkpoint via `MODEL_NAME` env variable
- **Live step-by-step visualisation** — Gradio dashboard streams each action, observation, and reward in real time as the agent plays the episode
- **HuggingFace Spaces deployed** — environment is live and publicly accessible without any local setup

---

## What the Environment Forces the Agent to Do

The agent starts every episode with one thing: an email inbox. What happens next depends entirely on what actions it chooses.

### Feature 1 — Multi-App Workflow (The Core Loop)

The agent cannot complete any task by reading a single source of truth. It must synthesize information across two separate simulated applications:

1. **Email Inbox** — unstructured natural language. The invoice arrives as a realistic email body with line items, prices, tax amounts, an IBAN bank account, and a vendor name embedded in prose.
2. **ERP Database** — a structured REST API (think SAP or Oracle). The agent must query it with the correct schema to fetch the matching Purchase Order and cross-reference every field.

**The multi-app floor is enforced in the reward function:** if the agent never queries the ERP, its maximum possible episode score is capped at **0.10**, regardless of how perfectly it reads the email. This is hardcoded into `grade_task()` in `app/tasks.py` and cannot be gamed. It forces genuine cross-system reasoning every single episode.

### Feature 2 — Schema Drift (The Hard Task)

On the `hard` task, the ERP API silently upgrades mid-session. The agent's first query — using `vendor_name` as the lookup key — returns:

```
SCHEMA DRIFT: API v2 requires vendor_tax_id. vendor_name is deprecated.
```

The agent must parse this error, infer the new schema from the message, and retry the query with `vendor_tax_id`. No hints. No retry logic built in. On top of this, the hard task also requires the agent to detect a **duplicate invoice number** by running `match_duplicate`, and flag a **tax rate inflation** from the correct 8-12% to a fraudulent 15%.

This is exactly how real enterprise ERP systems behave during upgrades. Any agent that cannot adapt produces a false `approved` and lets the payment through.

### Feature 3 — Expert Fraud Detection (The Ultimate Test)

The `expert_fraud` task injects two independent attack vectors that **must both be caught** for full score:

**Attack 1 — Lookalike Email Domain:** Each of the 5 vendors has a hand-crafted fraud domain baked into `app/tasks.py`:

| Legitimate Domain | Fraud Domain | Attack Technique |
|-------------------|-------------|------------------|
| `techsupplies.com` | `techsuppIies.com` | Capital-I substituting lowercase-L |
| `officemart.ltd` | `0fficemart.ltd` | Zero substituting letter-O |
| `globaltech.com` | `g1obaltech.com` | Numeral-1 substituting letter-L |
| `vertex.com` | `vertx.com` | Silent letter drop |
| `cloudserve.io` | `c1oudserve.io` | Numeral-1 substituting letter-L |

**Attack 2 — Fraudulent IBAN:** The invoice contains a bank account number that does not match the ERP vendor profile. This IBAN is **generated fresh every episode using Python's `secrets` module** (OS-level entropy) — it cannot be predicted by seeding any random number generator. The agent must catch the mismatch by comparing the invoice IBAN against the ERP's registered `iban` field.

The fraud email also creates urgency with a suspiciously short **10-day due date** (vs. standard 15–45 days) — a social engineering signal a trained agent should flag.

The agent must raise **both** `fraud` (domain mismatch) and `fraud_iban` (IBAN mismatch) flags to earn full score. Catching only one is partial credit.

### Feature 4 — Multi-Turn Vendor Negotiation

The `expert_negotiation` task tests something no standard benchmark does: **can an AI agent close a real business loop autonomously?**

The invoice arrives inflated by 10–25% — the vendor forgot to apply a pre-agreed partnership discount. The ERP shows the correct approved amount. The agent must:

1. Detect the discrepancy (invoice total ≠ ERP `approved_amount`)
2. Compose and send a `send_email` action to the vendor's sales address
3. **The environment's `simulated_responses` engine triggers:** it detects the `send_email` targeting the correct vendor email and dynamically injects a corrected invoice (`email_005`) directly into the inbox
4. Read the corrected email — which contains the vendor's apology and revised line items with the discount applied
5. Re-extract all 8 fields from the corrected invoice
6. **Approve** — only the corrected amounts are the ground truth; the original was wrong

### Feature 5 — Adaptive Curriculum Training

The training loop in `train.py` implements an **Adaptive Self-Improving Curriculum** that tracks a rolling 5-episode average per task:

- Score **> 88%** → automatically promoted to the next difficulty tier
- Score **< 50%** → demoted to rebuild competence

This ensures the agent's compute budget is spent exactly at the frontier of its current ability — not replaying tasks it has already mastered.

### Feature 6 — Procedural Generation (No Memorization Possible)

Every call to `reset()` generates a completely fresh episode. The generator randomly selects from:
- **5 vendors** with unique domains, fraud domains, tax IDs, IBAN numbers, and product catalogues
- **Tax rates:** 8%, 10%, 12%, or 15%
- **Invoice dates:** random across a 700-day window (2024–2026)
- **Due dates:** 10, 15, 30, or 45 days from invoice date
- **Line items:** 2–3 items randomly sampled from each vendor's catalogue, with quantities and prices randomized within per-item ranges
- **Fraud IBANs:** generated with `secrets.choice()` (OS entropy) — unseeded and unpredictable per episode

An agent cannot memorize answers. It must reason every time.

---

## The Reward Architecture: Why This Environment Actually Teaches

Most environments use binary scoring: win = 1, lose = 0. Binary rewards create sparse gradients — sparse gradients mean slow learning. We used **OpenEnv's Rubric system** to build a **dense, step-level reward function**:

| Action | Reward | Why |
|--------|--------|-----|
| `read_email` — correct email found | **+0.05** | Rewarded for navigating the inbox correctly |
| `query_erp` — vendor located in ERP | **+0.10** | Cross-system lookup completed |
| `extract` — field value matches ground truth | **+0.07** | Accurate extraction from unstructured text |
| `flag` — correct anomaly flagged | **+0.12** | Agent identified a genuine issue |
| `send_email` — vendor negotiation triggered | **+0.20** | Multi-turn resolution loop initiated |
| `match_duplicate` — duplicate invoice confirmed | **+0.12** | Resubmitted invoice caught |
| Incorrect action / wrong state | **−0.01 to −0.05** | Penalty proportional to severity |
| ERP never queried (any task) | **Hard floor: 0.10** | Multi-app requirement enforced unconditionally |

**The final episode score** is a weighted rubric across three components:

- **40%** — Field extraction accuracy (8 fields: `vendor_name`, `invoice_number`, `invoice_date`, `due_date`, `subtotal`, `tax_amount`, `total_amount`, `iban`)
- **30%** — Workflow compliance (correct flags raised, negotiation completed, or correct clean-invoice handling)
- **30%** — Final decision accuracy (`approve` vs. `reject` — must match `expected_decision` in the task spec)

A model that extracts data perfectly but makes the wrong final decision scores ~0.40. A model that makes the right decision without querying the ERP scores 0.10. **The reward function is not gameable by exploiting any single component.**

---

## The 5 Levels of Enterprise Complexity

| Level | Task | Key Challenge |
|-------|------|--------------|
| 1 | **Easy** | Read email → Query ERP → Extract 8 fields → Approve |
| 2 | **Medium** | Detect a 12–30% price discrepancy vs. the signed PO, flag and reject |
| 3 | **Hard** | ERP API silently upgrades: `vendor_name` deprecated, `vendor_tax_id` required. Also detect duplicate invoice + tax mismatch. |
| 4 | **Expert Negotiation** | Invoice is inflated. Email the vendor, receive corrected invoice dynamically injected into inbox, re-extract all 8 fields, approve. |
| 5 | **Expert Fraud** | Sender is a lookalike domain. IBAN is cryptographically random and doesn't match ERP. Flag both `fraud` + `fraud_iban`, reject. |

Each level introduces a failure mode that breaks standard automation. The hardest tasks require the agent to reason across multiple turns, multiple data sources, and multiple adversarial signals simultaneously.

---

## Proving the Gym Works: GRPO Reinforcement Learning

To validate that the environment produces a learnable gradient signal, we ran **GRPO (Group Relative Policy Optimization)** using **Unsloth + TRL** on a **free Google Colab T4 GPU**. Full loop in under 35 minutes.

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/llama-3-8b-Instruct-bnb-4bit` |
| Quantization | 4-bit NF4 (Unsloth) — ~5 GB VRAM |
| Fine-tuning method | LoRA (r=16, α=32) — ~1% of parameters trainable |
| RL algorithm | GRPO (`GRPOTrainer` from TRL) |
| Group size | 4 candidate responses per prompt |
| Training steps | 100 |
| Dataset size | 125 synthetic AP workflow state prompts |
| Reward function | AP environment rubric distilled from `grade_task()` |
| Hardware | Google Colab T4 GPU (free tier) |
| Runtime | ~35 minutes end-to-end |

**Why GRPO over PPO?** PPO requires a separate value-head and reference model, which doubles VRAM usage and OOMs on a free T4. GRPO computes advantage by comparing the model's own outputs within a group — no extra model needed.

**Why Unsloth?** Unsloth's custom CUDA kernels and 4-bit quantization reduce the 8B model from 16 GB → ~5 GB, leaving headroom for LoRA gradients.

### What Happened During Training

![GRPO Training Curves — Loss ↓ and Reward ↑ over 100 steps (Llama-3.1-8B + LoRA on Colab T4)](./grpo_training_curves.png)

*Left panel: Training loss drops from ~0.0025 to a final of **0.001** — the model is consistently improving. Right panel: Average reward per step climbs from below 0.2 and stabilises around 0.48–0.65, approaching the 0.70 target threshold (dashed line). The smoothed green trend confirms a genuine upward learning signal — not noise.*

### Before vs. After GRPO — The Result

![Enterprise AP-Env: RL-Trained Model Improvement — Llama-3.1-8B + Unsloth GRPO](./rl_training_results_final.png)

*Easy Task: **0.26 → 0.88** (+0.62). Expert Negotiation: **0.18 → 0.78** (+0.60). Both tasks now exceed the 0.70 pass threshold. Peak reward nearly tripled on Easy. The model learned when to `approve` a clean invoice instead of hallucinating flags.*

| Task | Before Training | After GRPO | Improvement |
|------|----------------|------------|-------------|
| **Easy** | 0.26 | **0.88** | **+0.62 (+238%)** |
| **Expert Negotiation** | 0.18 | **0.78** | **+0.60 (+333%)** |

**The reward curve goes up. The loss curve goes down. The gym works. The signal is real.**

---

## Full Tech Stack: 

| Tool | Role | Why We Chose It |
|------|------|-----------------|
| **OpenEnv** (`openenv-core`) | Gym base class + `openenv.yaml` spec | Standard interface — any agent can plug in instantly via Gym-style API |
| **FastAPI** | REST simulation server (`app/main.py`) | High-performance, async-ready, auto-generates `/docs` for judges |
| **Pydantic v2** | Action & observation schema validation (`app/models.py`) | Strict type enforcement on every agent action at runtime |
| **Gradio** | Live visual dashboard (`app/demo.py`) | Judges can **watch** the agent play episodes in real-time on HF Spaces |
| **Uvicorn** | ASGI server | Production-grade concurrency for multi-agent training |
| **Any HF model** (we used `Llama-3.1-8B-Instruct`) | LLM agent via HF Router API | The environment is model-agnostic — plug in any HuggingFace-compatible LLM or your own fine-tuned model |
| **Unsloth** | 4-bit quantized model loading + CUDA kernels | 8B model VRAM: 16 GB → 5 GB; enables free T4 training |
| **TRL (GRPOTrainer)** | RL training loop | No reference model needed — GRPO fits in a single T4 |
| **LoRA** (via `peft`) | Efficient fine-tuning adapters | ~1% of parameters trained — full model quality at fraction of compute |
| **HuggingFace Hub** | Model hosting + Space deployment | Free GPU hosting for the live demo |
| **HuggingFace Router API** | LLM inference for `inference.py` | Provider-agnostic — swap models without changing code |
| **Matplotlib** | All result charts | Generated directly from real run JSON files |
| **Python `secrets` module** | Fraud IBAN generation | OS-level entropy — fraud IBANs are truly unpredictable per episode |
| **Google Colab T4** | Training hardware | Anyone can reproduce our results for free |

---

## Technical Architecture

```
enterprise-ap-env/
├── app/
│   ├── main.py              # FastAPI server — UUID sessions, TTL eviction, thread-safe
│   ├── environment.py       # OpenEnv state machine — reset() / step() / state
│   ├── tasks.py             # 5 procedural task generators + grade_task() rubric
│   ├── models.py            # Pydantic schemas: Action, Observation, StepResult
│   └── demo.py              # Gradio live dashboard — rule-based + LLM agent modes
├── train.py                 # Adaptive curriculum training loop (reference agent)
├── inference.py             # Headless LLM agent — HF Router API, any model
├── plot_llm_results.py      # Reward curve visualization from JSON results
├── colab_rl_training.ipynb  # Unsloth GRPO training notebook (T4-optimized)
├── eval_after_training.py   # Before/after evaluation script (Colab-compatible)
├── openenv.yaml             # OpenEnv environment specification
└── Dockerfile               # HuggingFace Spaces deployment
```

### FastAPI Backend — Thread-Safe Multi-Agent Server

- **UUID-based session management** — every episode gets a unique isolated session
- **Idle-timeout garbage collection** — sessions auto-expire, preventing memory leaks during multi-agent training
- **Thread-safe state isolation** — multiple LLMs can train simultaneously on the same server without state collisions
- **OpenAI-compatible REST interface** — `/reset`, `/step`, `/state`, `/health`, `/docs`
- **Fully decoupled architecture** — the FastAPI server, headless benchmarking script (`inference.py`), and Gradio dashboard can each run independently

Any agent framework — LangChain, AutoGPT, raw `requests.post()` — can plug in.

### Gradio Dashboard — 

`app/demo.py` is not just a demo. It is an observability tool. Watch live: inbox loads, email opens, ERP response appears, fields extract one by one, a flag raises, final decision renders.

**The dashboard has two agent modes — it switches automatically based on whether you provide an HF token:**

| Mode | How to activate | Who runs the episode |
|------|----------------|----------------------|
| **Scripted Agent** (default) | Open the Space — no token needed | A deterministic rule-based agent plays the episode automatically. Always works, no API calls. |
| **Autonomous LLM Agent** | Paste your HuggingFace token in the sidebar and set the model name, then click Run | Any HuggingFace-compatible LLM drives the episode live via the Router API — Llama, Mistral, Qwen, your own fine-tuned model, anything. Every action is generated by the model in real time. |

No token → scripted reference agent plays. Token + model name provided → your chosen LLM takes over. The environment, reward function, and observation format are identical in both modes — the only difference is who is making the decisions.

> This means any researcher can plug in their own trained model and immediately benchmark it against the same 5 tasks, the same reward rubric, and the same adversarial conditions — without writing a single line of integration code.

### OpenEnv — The Standard Interface

The environment inherits from `openenv-core`'s `Environment` base class and exposes the standard `reset()`, `step()`, `state` API. The `openenv.yaml` specification declares all 5 tasks, their action/observation spaces, and the full reward rubric — researchers can understand the environment without reading source code.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset?task_name=easy` | POST | Start a new episode — returns unique `session_id` |
| `/step?session_id=<uuid>` | POST | Take one action |
| `/state?session_id=<uuid>` | GET | Inspect live state of active session |
| `/sessions` | GET | List all active sessions |
| `/tasks` | GET | List all available tasks |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

---

## Quick Start

```bash
pip install -r requirements.txt

# Start the environment server
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860

# Run the Gradio live demo (no server needed)
python app/demo.py

# Run adaptive curriculum training
python train.py --episodes 60

# Benchmark any LLM (set HF_TOKEN first)
export HF_TOKEN="hf_your_token_here"
python inference.py --all --episodes 5

# Plot reward curves
python plot_llm_results.py
```

**GRPO Training on Colab:** Open `colab_rl_training.ipynb` → Runtime → T4 GPU → Run All. Completes in ~35 minutes.

---


We open-sourced everything.

**Enterprise is production-grade infrastructure for a $2.9 billion real-world problem.**

---

*Built at the Meta AI OpenEnv Hackathon Grand Finale, April 2026.*
*Team SHIPWithTEA: Prachi & Dharmendra*

*Stack: OpenEnv · FastAPI · Pydantic · Gradio · Uvicorn · Unsloth · TRL (GRPO) · LoRA · HuggingFace Hub · Llama-3.1-8B-Instruct · Matplotlib · Python secrets · Google Colab T4*
