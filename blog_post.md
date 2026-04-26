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

#### Real LLM Validation (Llama-3.1-8B via HuggingFace API)

A real LLM agent (`meta-llama/Llama-3.1-8B-Instruct`) was driven against the live FastAPI server using `inference.py` with no task-specific hardcoding. Results from single-episode runs:

| Task | Score | Outcome |
|------|-------|---------|
| Easy | 0.99 | PASS — all 8 fields extracted, correct approval |
| Expert Fraud | 0.99 | PASS — detected `billing@vertx.com` lookalike, flagged `fraud` + `fraud_iban`, rejected |

The environment is live and LLM-accessible via the OpenAI-compatible REST interface.

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
