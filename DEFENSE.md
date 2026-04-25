# Judge Q&A Defense Cheatsheet
### Enterprise AP-Env — Meta AI Hackathon Grand Finals

Read this on the way to the venue. These are the three attacks you will face.

---

## 🔴 Attack 1 — "This isn't really multi-agent."

**What they'll say:**
> "You claim multi-agent interaction, but Task 4 is just a scripted dict that appends an email
> to a list. That's not multi-agent, that's an if-statement."

**Your answer:**
> "You're right that we don't have two independently-running LLM processes communicating.
> What we deliberately built is **multi-turn environment dynamics with a simulated reactive actor**.
> The research question we're answering is: *can an agent detect a hidden information gap,
> decide to solicit that information, and then incorporate the new data into its reasoning?*
> That's the cognitive behavior we care about benchmarking — the architecture of the vendor
> simulation is secondary to the agent's decision-making challenge. We called it out accurately
> in our OpenEnv spec."

**Why this works:** You're owning the scope, not defending a wrong claim. Researchers respect honesty about scope far more than bluster.

---

## 🟡 Attack 2 — "Enterprise AP is a vision/OCR problem. You're giving the agent clean text."

**What they'll say:**
> "Real invoices are PDFs, scanned images, handwritten notes. Your environment feeds the agent
> a perfectly formatted text string. You've abstracted away the hardest part."

**Your answer:**
> "Correct — and that's intentional. We isolated the **cognitive reasoning and system-interaction
> layer** from the perception layer. The hardest unsolved problem in agentic AP isn't OCR —
> it's what the agent does once it *has* the data: adapting to schema drift, detecting
> character-level fraud, resolving pricing disputes across multi-step workflows.
> That's the layer we're benchmarking. Vision parsing sits below this environment as a
> pre-processing step, and plugging in a vision model is a one-line change to how the
> email body is populated."

**Why this works:** You frame the scope decision as deliberate engineering, not a gap. And the "one-line change" claim is true — `_invoice_body()` in `tasks.py` is the only place to swap in OCR output.

---

## 🟡 Attack 3 — "Dense reward shaping causes reward hacking."

**What they'll say:**
> "Your +0.07 per field, +0.12 per flag system will train an agent to grind through
> low-hanging rewards without actually solving the task. That's reward hacking."

**Your answer:**
> "We specifically designed against this with two safeguards. First, the `grade_task()` function
> is a **hard-gating sparse signal** — if the agent never queries the ERP, the maximum
> final score is 0.10, regardless of how many step rewards it accumulated. Second, the
> final score is 40% field accuracy, 30% correct flags, 30% correct decision — all computed
> against ground truth, not against what actions were taken. An agent that extracts wrong
> values gets zero for that component. The dense step rewards exist purely to give the agent
> a non-zero gradient during early training on multi-step LLM chains — without them, the
> credit assignment problem across 10–15 steps becomes intractable."

**Why this works:** You demonstrate you understand RL theory (sparse reward credit assignment) and that you anticipated the problem.

---

## ⚡ Bonus — "How is this different from LangChain benchmarks?"

> "LangChain benchmarks test **tool retrieval** — can the agent call the right function?
> Our environment tests **multi-step financial reasoning with consequential decisions**:
> the agent must integrate data from two disconnected apps, detect silent API changes,
> handle malicious inputs, and make a binary approve/reject call — all scored against
> real ground truth. The cost of a wrong decision is explicit in the reward signal."

---

## 🎯 Opening 30 Seconds — Lead With This

Don't start with Task 1. Say:

> *"Last year, companies lost $26 billion to invoice fraud.
> We built the environment that teaches any AI to catch it —
> and we're open-sourcing it so the whole community can run their agents against it."*

Then go straight to **Task 5 (Expert Fraud)**. Show the lookalike domain. Let the visual do the talking.

---

## 📋 Key Numbers To Have Ready

| Fact | Value |
|------|-------|
| Sessions are isolated by | UUID `session_id` |
| Idle session eviction after | 300 seconds (5 min) |
| Smoke tests passing | 8/8 |
| Reward structure | 40% fields / 30% flags / 30% decision |
| Noise decay range | 0.55 → 0.00 over 50 episodes |
| ERP hard floor (if never queried) | 0.10 max score |
| Fraud vendor domains | 5 unique lookalike patterns |
| Tax rates in play | 8%, 10%, 12%, 15% |
| LLM score on easy + fraud | 0.99 both |

---

*Built for Meta AI Hackathon Grand Finals 2026 — Dharmendra & Prachi*
