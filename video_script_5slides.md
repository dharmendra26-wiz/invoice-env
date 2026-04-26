# Enterprise AP-Env — 5-Slide PPT & Video Script
## Meta AI Hackathon Grand Finale 2026
*Target runtime: 2 minutes 45 seconds – 3 minutes. Read at a natural, slightly unhurried pace.*

---

## SLIDE 1 — Title Slide

### Visual Design
- **Headline:** `Enterprise AP-Env`
- **Subtitle:** `A Real-World Benchmark for Agentic Workflows`
- **Sub-line:** `Meta AI OpenEnv Hackathon · Grand Finale 2026`
- **Team:** Prachi & Dharmendra
- **Graphic:** Center-right — a robot in a business suit sitting at a desk, reading a paper invoice with a magnifying glass. Background: dark corporate aesthetic (navy/charcoal). Bottom strip: OpenEnv + Hugging Face + Unsloth logos.

### Audio / Subtitles
> "Current open-source environments train agents to play chess, navigate mazes, or solve grid-worlds.
>
> But if we want AI to actually work in the real world — in the places where real money moves — we need to train them on real-world risk.
>
> Welcome to Enterprise AP-Env.
>
> This is a high-stakes Accounts Payable simulation built for the Meta Hackathon. An environment where the agent isn't moving a game piece — it's processing invoices, catching fraudsters, and negotiating with vendors.
>
> And if it makes a mistake, it doesn't just lose points. It authorizes a wire transfer to a criminal."

*Timing: ~28 seconds*

---

## SLIDE 2 — The Environment Features

### Visual Design
- **Headline:** `What the Agent Must Survive`
- **Layout:** 4 feature cards arranged in a 2×2 grid, each with an icon + title + 1-line description:

| Card | Icon | Title | One-liner |
|------|------|-------|-----------|
| Top-left | 📧+🗃️ | Multi-App Workflow | Read the email. Query the ERP. No single source of truth. |
| Top-right | ⚡ | Schema Drift | ERP API upgrades mid-session. Adapt or fail. |
| Bottom-left | 🕵️ | Expert Fraud Detection | Lookalike domain? Mismatched IBAN? Flag both or lose. |
| Bottom-right | ✉️ | Multi-Turn Negotiation | Email the vendor. Wait for a reply. Close the loop. |

- **Small note at bottom:** "5 tasks · 5 vendors · procedurally generated every episode — no memorization possible"

### Audio / Subtitles
> "We built a multi-app sandbox where an agent must act as a financial controller.
>
> First, it reads a vendor email — unstructured natural language, line items, IBAN bank accounts, tax rates.
>
> Then it queries a live ERP database to cross-reference the Purchase Order. If it never queries the ERP, its maximum possible score is capped at 0.10. No exceptions.
>
> On the harder tasks, the ERP API silently upgrades. The lookup key changes from vendor_name to vendor_tax_id. The agent must read the error, infer the new schema, and retry.
>
> On the fraud task, the invoice comes from a lookalike domain — like techsuppIies.com — with a capital letter I instead of a lowercase L. And the bank account is a randomly generated IBAN that doesn't match the vendor's ERP profile.
>
> On the negotiation task, the agent can't just reject. It must email the vendor, receive a corrected invoice, and then approve.
>
> It's not just text extraction. It's a tool-use survival course."

*Timing: ~52 seconds*

---

## SLIDE 3 — The Reward Engine

### Visual Design
- **Headline:** `A Granular Reward Signal — Not Binary Win/Lose`
- **Layout:** A vertical flowchart showing the agent's workflow, with reward amounts annotated at each step:

```
[ Read Email ]          → +0.05
       ↓
[ Query ERP ]           → +0.10  (floor 0.10 if skipped)
       ↓
[ Extract each field ]  → +0.07 per correct field (8 fields)
       ↓
[ Flag anomaly ]        → +0.12 per correct flag
       ↓
[ Send vendor email ]   → +0.20 (negotiation loop)
       ↓
[ Approve / Reject ]    → Final rubric:
                           40% Extraction Accuracy
                           30% Workflow Compliance
                           30% Decision Accuracy
```

- **Bottom note:** "Prevents reward hacking — you can't score high by exploiting a single component"

### Audio / Subtitles
> "To make this a true training ground, we used OpenEnv to design a highly granular reward rubric.
>
> We don't give a 1 for winning and a 0 for losing.
>
> The agent earns plus 0.05 for opening the right email. Plus 0.10 for a successful ERP query. Plus 0.07 for every correct field extracted from the invoice. Plus 0.12 for correctly flagging an anomaly like a mismatched IBAN. And plus 0.20 for completing the multi-turn vendor negotiation.
>
> The final score is computed across three components: 40 percent for field extraction accuracy, 30 percent for workflow compliance, and 30 percent for the correct final decision.
>
> This means a model that perfectly extracts the data but approves a fraudulent invoice still scores below 0.45. The reward function is not gameable by exploiting any single step.
>
> This creates a rich, informative gradient for reinforcement learning — exactly what GRPO needs to work."

*Timing: ~50 seconds*

---

## SLIDE 4 — The Proof (Stress-Testing with GRPO)

### Visual Design
- **Headline:** `We Proved the Gym Works`
- **Layout:** Side-by-side panels

**Left panel — The Baseline:**
- Title: `Before GRPO Training`
- Shows the Llama-3.1-8B baseline curve (flat/near-zero line)
- Caption: *"Base Llama 3.1 8B — extracts data perfectly, then hallucinates flags on clean invoices and never issues approve. Scores 0.10 on Easy after 25 steps."*

**Right panel — After Training:**
- Title: `After GRPO on Colab T4 (~35 min)`
- Shows the before/after bar chart (green bars significantly higher)
- Key numbers annotated: Easy: 0.26 → 0.88 · Negotiation: 0.18 → 0.78
- Caption: *"Peak reward nearly tripled. Loss consistently decreasing. The environment is learnable."*

### Audio / Subtitles
> "To prove our environment actually teaches models, we stress-tested it.
>
> We took a base Llama 3.1 8B model and ran it through 25 episodes on our live API server. No fine-tuning, no prompting tricks.
>
> The result? On the Easy task, the model extracted all 8 invoice fields perfectly — and then panicked. It started hallucinating flags on a clean invoice and never issued an approve action within the step limit. It scored 0.10.
>
> That's the baseline. That's what happens when you deploy an unspecialized model into a real corporate workflow.
>
> Then we ran GRPO. Unsloth 4-bit quantization, LoRA adapters, 100 training steps on a free Colab T4 GPU. The full loop ran in under 35 minutes.
>
> After training, the Easy task score climbed from 0.26 to a peak of 0.88. Expert Negotiation went from 0.18 to 0.78.
>
> The reward curve goes up. The loss curve goes down. The environment is learnable. The gym works."

*Timing: ~52 seconds*

---

## SLIDE 5 — Live Demo & Tech Stack

### Visual Design
- **Headline:** `Try It Right Now`
- **Center:** Embedded video or animated GIF of the Gradio dashboard running a live episode — the inbox loads, an email opens, ERP response appears, fields populate one by one, a flag is raised, the agent approves.
- **Bottom strip — 4 logos with captions:**

| Logo | Caption |
|------|---------|
| OpenEnv | Gym wrapper + composable rubric scoring |
| Hugging Face | Live Space + model card + training notebook |
| Unsloth | 4-bit quantized RL in under 35 minutes |
| Gradio | Visual agent observability dashboard |

- **QR code** bottom-right: links to the HF Space

### Audio / Subtitles
> "This is the Gradio dashboard running a live episode. Watch: the inbox loads, the agent reads the email, queries the ERP, extracts each field one by one, raises a fraud flag on the mismatched IBAN, and rejects the payment.
>
> You can run this right now — the Space is live on Hugging Face.
>
> We built the environment with OpenEnv for standards compliance. The UI with Gradio so judges can watch the agent think in real time. And we proved the RL pipeline with Unsloth and TRL's GRPO — running a full training loop on a free T4 GPU.
>
> Every component is open. Every result is real. Every invoice is procedurally generated — no memorization possible.
>
> Enterprise AP-Env doesn't just evaluate models. It teaches them how to do business.
>
> Thank you."

*Timing: ~45 seconds*

---

## Full Script Timing Summary

| Slide | Topic | Target Time |
|-------|-------|-------------|
| 1 | Title & Hook | 0:00 – 0:28 |
| 2 | Environment Features | 0:28 – 1:20 |
| 3 | Reward Engine | 1:20 – 2:10 |
| 4 | The Proof | 2:10 – 3:02 |
| 5 | Live Demo & Stack | 3:02 – 3:47 |

*Total: ~3 minutes 47 seconds at a comfortable pace. Trim slide 2 or 4 by 30s if you need to hit exactly 2:45.*

---

## Production Notes

- **Recording setup:** Use OBS Studio or Loom. Record at 1080p. Use your system microphone with a pop filter or a headset — avoid laptop mics.
- **Slide transitions:** Fade, 0.3s. Do not use flying animations — they look unprofessional and waste time.
- **Subtitle format:** Export as SRT or burn-in with CapCut / DaVinci Resolve. White text, dark shadow, bottom-center.
- **Slide 4 visual tip:** Use the `Llama_3.1_8B_Instruct_real_curves.png` on the left panel and `before_after_results.png` on the right panel.
- **Slide 5 visual tip:** Start the Gradio UI running an `expert_fraud` episode — it's the most visually dramatic (URGENT subject line, fraud flag, IBAN mismatch, reject).
- **Background music:** Optional. If used, keep at -20 dB underneath voice. Lo-fi corporate or minimal electronic.
