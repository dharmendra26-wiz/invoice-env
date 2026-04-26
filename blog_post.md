# Enterprise AP Environment: A New Benchmark for Financial AI Agents

*(Built for the Meta AI Hackathon — Grand Finale 2026)*

---

### The $2.9 Billion Problem

Every 20 seconds, a corporation wires money to a criminal. Not because their accounting team is careless, but because the fraud was designed to look identical to a real invoice. Business Email Compromise (BEC) and invoice fraud cost enterprises over **$2.9 billion per year**.

The AI industry races to deploy Large Language Models to automate Accounts Payable workflows, but there is a fundamental problem: **there is no standard benchmark to test whether these agents are actually safe.** Existing benchmarks like MMLU or SWE-bench test trivia or code, not multi-application corporate workflows under adversarial conditions. You cannot trust an AI with your treasury if you have not stress-tested it against fraud, schema drift, and price manipulation.

We built the measurement. Introducing the **Enterprise AP Environment**.

---

### Try It Live

* **Live Interactive Demo:** [https://huggingface.co/spaces/decent-cow26/invoice-env](https://huggingface.co/spaces/decent-cow26/invoice-env)
* **HuggingFace Model Card & Writeup:** [https://huggingface.co/Prachi-2601/Multi-App-RL-Env-Invoice-Processing-Schema-Drift-Fraud-Detection-Vendor-Negotiation](https://huggingface.co/Prachi-2601/Multi-App-RL-Env-Invoice-Processing-Schema-Drift-Fraud-Detection-Vendor-Negotiation)

---

### Key Engineering Features

* **Fully Decoupled Architecture:** A FastAPI simulation server, a headless benchmarking script (`inference.py`), and a Gradio visual dashboard — each independently operable.
* **OpenEnv Compliant:** Inherits the `openenv-core` base class. Any researcher can plug their LLM into the environment instantly via the standard Gym-style API.
* **Procedural Generation:** Vendor names, prices, tax rates, dates, and fraudulent bank accounts are dynamically generated every episode using a 5-vendor pool. The AI cannot memorise answers — it must reason every time.
* **Thread-Safe Session Management:** UUID-based sessions with idle-timeout garbage collection allow multiple agents to train simultaneously on the same server without state collisions.
* **Schema Drift Simulation:** The ERP API silently changes its required field mid-session, forcing the agent to read an error message, infer the new schema, and retry.
* **Adaptive Self-Improving Curriculum:** The environment automatically promotes or demotes the agent's difficulty tier based on its rolling 5-episode performance, implementing Theme #4 directly in the training loop.

---

### The Architecture

Think of this as the OpenAI Gym for enterprise finance. The agent is dropped into a simulated AP department terminal where it must:

1. **Read** incoming vendor emails from an inbox.
2. **Query** a simulated ERP database (like SAP or Oracle) using the correct API schema.
3. **Extract** structured financial data from unstructured email text.
4. **Negotiate** price discrepancies autonomously over simulated vendor email.
5. **Detect** sophisticated phishing domains and IBAN bank fraud.

Every action is rewarded immediately (step-level shaping), and the episode ends with a structured final score across three components: Extraction Accuracy (40%), Workflow Flags (30%), and Final Decision (30%).

---

### The 5 Levels of Enterprise Complexity

| Level | Task | Key Challenge |
|-------|------|--------------|
| 1 | **Easy** | Read email → Query ERP → Extract 8 fields → Approve |
| 2 | **Medium** | Detect a 12–30% price discrepancy vs. the signed PO |
| 3 | **Hard** | ERP API silently upgrades: `vendor_name` deprecated, `vendor_tax_id` required. Detect duplicate invoice. |
| 4 | **Expert Negotiation** | Autonomously email vendor about inflated price, receive corrected invoice injected into inbox, re-extract and approve |
| 5 | **Expert Fraud** | Sender is `@techsuppIies.com` (capital-I lookalike). IBAN is cryptographically random and does not match ERP records. Flag both and reject. |

Each level introduces a failure mode that breaks standard automation systems. The fraud IBAN is generated using Python's `secrets` module (OS-level entropy) so it cannot be pre-computed by seeding the random number generator.

---

### Benchmark Results

#### Reference Agent Validation (100 Episodes)

To validate environment correctness and demonstrate discriminative power, we ran a deterministic rule-based reference agent at two competency levels — **High Noise (simulating a weak model)** and **Low Noise (simulating a strong model)** — across all 5 tasks for 20 episodes each.

| Task | Weak Agent (High Noise) | Strong Agent (Low Noise) | Gap |
|------|------------------------|--------------------------|-----|
| Easy | 82% | 94% | +12 pts |
| Medium | 83% | 95% | +12 pts |
| Hard | 77% | 95% | +18 pts |
| Expert Negotiation | 79% | 96% | +17 pts |
| Expert Fraud | 65% | 86% | +21 pts |
| **Average** | **77%** | **93%** | **+16 pts** |

The environment reliably separates weak and strong agents. The hardest tasks (Expert Fraud, Hard) show the largest gaps — exactly what a well-designed benchmark should do.

![Reference Agent Reward Curves — Weak (8B noise) vs Strong (70B noise)](./8B_curves.png)

#### Real LLM Benchmark — Llama-3.1-8B-Instruct (25 Live Episodes, 5 per task)

A real `meta-llama/Llama-3.1-8B-Instruct` agent was driven against the live FastAPI server using `inference.py` with zero task-specific hardcoding. Every episode used a procedurally generated, unique invoice.

| Task | Avg Score | Result | Key Behaviour |
|------|-----------|--------|---------------|
| Easy | **0.10** | FAIL | Correctly extracts all 8 fields and queries ERP, but then hallucinates spurious flags (`price_mismatch`, `fraud_iban`, etc.) on a clean invoice — never issues `approve` within 25 steps |
| Medium | **0.99** | PASS | Price mismatch detected and invoice rejected in all 5 episodes, in 13–14 steps |
| Hard | **0.99** | PASS | Schema drift handled perfectly — v1 ERP rejected, model auto-retries with `vendor_tax_id` on v2 |
| Expert Negotiation | **0.23** | FAIL | Model correctly identifies the price discrepancy and sends a vendor email, but cannot close the multi-turn negotiation loop before the step limit |
| Expert Fraud | **0.88** | PASS | Fraudulent IBAN flagged and payment rejected in all 5 episodes; 1/5 episodes also caught the lookalike domain |
| **Average** | **0.64** | — | — |

![Real LLM Reward Curves — Llama-3.1-8B-Instruct, 5 episodes per task](./Llama_3.1_8B_Instruct_real_curves.png)

**What this reveals about the environment's discriminative power:**

The benchmark produces a capability profile, not a single number. The 8B model excels at pattern-detection and schema-recovery tasks (Hard: 0.99, Medium: 0.99, Fraud: 0.88) but fails at two distinct failure modes:

1. **Workflow adherence (Easy: 0.10):** The model extracts all 8 fields perfectly and queries the ERP correctly, then gets stuck in a loop of hallucinated flags on a clean invoice. It "knows too much" — it keeps looking for a problem that isn't there. GRPO training on the `approve` state specifically addresses this failure.

2. **Multi-turn planning (Negotiation: 0.23):** The model recognises the price discrepancy and correctly initiates vendor contact, but cannot read the updated inbox, re-extract fields from the corrected invoice, and close with an `approve` action — all within the step budget. This is precisely the multi-turn planning task that no standard benchmark tests.

This is exactly the kind of signal a useful benchmark should produce — not a single accuracy number, but a precise capability profile that tells a researcher where the model breaks down under enterprise conditions.

---

### GRPO Reinforcement Learning Training

The environment is not just a benchmark — it is a training signal. `colab_rl_training.ipynb` trains `meta-llama/Llama-3.1-8B-Instruct` (4-bit quantized via Unsloth) directly on the AP environment's reward logic using **GRPO** (Group Relative Policy Optimization) from HuggingFace TRL.

**Why GRPO and not PPO?** PPO requires a separate reference model and value head, which doubles VRAM usage and is unstable for short training runs. GRPO generates a group of candidate responses per prompt, ranks them by reward, and updates the model to prefer the better ones — no reference model required. It runs on a free T4 GPU in under 35 minutes.

**Training setup:**
- Base model: `unsloth/llama-3-8b-Instruct-bnb-4bit` (4-bit, ~5 GB VRAM)
- LoRA adapters (r=16) on all projection layers — only 1% of parameters are trainable
- 4 candidate actions generated per AP state prompt
- Reward function identical to `environment.py` step logic
- 125 training prompts spanning all 5 AP workflow stages
- 100 training steps (~25 min on T4)

The reward function scores each JSON output against the AP environment logic:

| Action quality | Reward |
|----------------|--------|
| Invalid JSON / hallucination | 0.0 |
| Wrong action for this workflow state | 0.1 |
| Correct action type, missing required fields | 0.5 |
| Correct action + all required fields | 1.0 |

Run the training yourself: open `colab_rl_training.ipynb` in Google Colab (T4 GPU), run all cells. You will see the reward curve trend upward and the loss curve trend downward within 100 steps.

---

### Adaptive Curriculum Training

Standard RL wastes episodes running agents on tasks they have already mastered. Our `train.py` implements an **Adaptive Self-Improving Curriculum** that tracks a 5-episode rolling average:

- If the agent scores **> 88%**: automatically promoted to the next difficulty tier.
- If the agent scores **< 50%**: safely demoted to rebuild competence.

This means the agent spends compute only on tasks at the frontier of its current ability — the hardest tasks it has not yet solved. In validation runs, this pacing ensures the agent reaches Expert Fraud within a fixed episode budget rather than plateauing on Easy.

---

### Why It Matters

Before a Fortune 500 company gives an autonomous AI agent access to their payment systems, they need proof that the agent will not wire $1 million to a lookalike domain. By building a thread-safe, OpenEnv-compliant, procedurally-generated environment, we have created the exact infrastructure that AI research labs and enterprise compliance teams need — and that does not yet exist anywhere else.

The Enterprise AP Environment is not a toy. It is a production-grade benchmark for a $2.9 billion real-world problem.
