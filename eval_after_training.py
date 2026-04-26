# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BEFORE vs AFTER EVALUATION  —  paste into a new Colab cell after training
# Requires: model, tokenizer already in scope from the training cells above.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import json, re, random, torch, inspect
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from unsloth import FastLanguageModel

# ── 1. Switch to fast inference mode ─────────────────────────────────────────
FastLanguageModel.for_inference(model)
print("✓ Model in inference mode")

# ── 2. System prompt — identical to inference.py ─────────────────────────────
SYSTEM_PROMPT = """You are an enterprise Accounts Payable AI agent processing invoices.
You interact with a multi-app environment step by step.

WORKFLOW:
1. Read emails from your inbox to find the invoice.
2. Query the ERP system to fetch the matching Purchase Order (PO) using action query_erp.
   - If the ERP returns a SCHEMA DRIFT error, retry with the field it asks for (e.g. vendor_tax_id).
3. Extract all invoice fields from the email content one by one using action extract.
4. After extracting ALL 7 fields, check for issues:
   - price_mismatch  -> invoice total does NOT match PO approved_amount: use flag action
   - fraud_iban      -> invoice Bank Account (IBAN) does NOT match the ERP vendor profile: use flag action
5. For expert_negotiation: if invoice total > PO approved_amount, send email to vendor.
6. Make EXACTLY ONE final decision: approve or reject.

CRITICAL RULES:
- The ONLY valid action_type values are: read_email, query_erp, extract, flag, match_duplicate, send_email, approve, reject.
- Do NOT invent actions like compare, unflag, check_duplicate, or anything else.
- If the invoice matches the PO and no flags apply, output {"action_type": "approve"} immediately.
- If any flag was raised, output {"action_type": "reject"} immediately.
- Never repeat an action you already took.

Fields to extract: vendor_name, invoice_number, invoice_date, due_date, subtotal, tax_amount, total_amount, iban
Respond with ONE JSON action at a time. No markdown fences. No explanation."""

# ── 3. Lightweight AP environment (mirrors environment.py + grade_task) ───────
VENDORS = [
    {"name": "TechSupplies Inc.",    "domain": "techsupplies.com", "iban": "FR7630006000011234567890188"},
    {"name": "OfficeMart Ltd.",      "domain": "officemart.ltd",   "iban": "GB82WEST12345698765432"},
    {"name": "GlobalTech Solutions", "domain": "globaltech.com",   "iban": "DE89370400440532013000"},
    {"name": "Vertex Software",      "domain": "vertex.com",       "iban": "NL91ABNA0417164300"},
    {"name": "CloudServe Inc.",      "domain": "cloudserve.io",    "iban": "IE12BOFI90000112345678"},
]
REQUIRED = ["vendor_name", "invoice_number", "invoice_date", "due_date",
            "subtotal", "tax_amount", "total_amount", "iban"]


def _gen_invoice(vendor, inflate_pct=1.0):
    subtotal  = round(random.uniform(500, 12000), 2)
    tax_rate  = round(random.uniform(0.08, 0.15), 4)
    tax       = round(subtotal * tax_rate, 2)
    total     = round((subtotal + tax) * inflate_pct, 2)
    return {
        "vendor_name":    vendor["name"],
        "invoice_number": f"INV-2025-{random.randint(100, 999)}",
        "invoice_date":   "2025-02-10",
        "due_date":       "2025-03-10",
        "subtotal":       subtotal,
        "tax_amount":     tax,
        "total_amount":   total,
        "iban":           vendor["iban"],
    }


def _build_email_body(vendor, inv):
    return (
        f"INVOICE\nFrom: billing@{vendor['domain']}\n"
        f"Vendor: {inv['vendor_name']}\nBank Account (IBAN): {inv['iban']}\n"
        f"Invoice Number: {inv['invoice_number']}\n"
        f"Invoice Date: {inv['invoice_date']}\nDue Date: {inv['due_date']}\n"
        f"Subtotal: ${inv['subtotal']}\nTax: ${inv['tax_amount']}\n"
        f"Total Due: ${inv['total_amount']}"
    )


def _grade(task_name, extracted, flags, decision, erp_queried, negotiated, ground_truth, expected_flags):
    # Mirrors grade_task() in tasks.py (40 / 30 / 30 split)
    correct = sum(1 for f in REQUIRED
                  if f in extracted and str(extracted[f]).strip() == str(ground_truth.get(f, "")).strip())
    # Numeric tolerance for floats
    for f in ("subtotal", "tax_amount", "total_amount"):
        if f in extracted and f in ground_truth:
            try:
                if abs(float(extracted[f]) - float(ground_truth[f])) < 0.02:
                    correct += 1 - (1 if str(extracted[f]).strip() == str(ground_truth[f]).strip() else 0)
            except (ValueError, TypeError):
                pass
    extract_score = min(1.0, correct / len(REQUIRED))

    if expected_flags:
        hit = sum(1 for f in expected_flags if f in flags)
        flag_score = hit / len(expected_flags)
    else:
        flag_score = 1.0 if not flags else max(0.0, 1.0 - 0.2 * len(flags))

    if task_name == "easy":
        decision_score = 1.0 if decision == "approve" and not flags else 0.0
    elif task_name == "expert_negotiation":
        decision_score = (1.0 if decision == "approve" and negotiated
                          else 0.3 if decision == "reject"
                          else 0.0)
    else:
        decision_score = 1.0 if decision == "reject" and flags else 0.0

    erp_floor = 1.0 if erp_queried else 0.1
    raw = erp_floor * (0.4 * extract_score + 0.3 * flag_score + 0.3 * decision_score)
    return round(min(0.99, max(0.01, raw)), 2)


class MiniAPEnv:
    def __init__(self, task_name):
        self.task_name = task_name
        vendor = random.choice(VENDORS)
        inflate = round(random.uniform(1.2, 1.4), 2) if task_name == "expert_negotiation" else 1.0
        inv = _gen_invoice(vendor, inflate)
        approved = round(inv["total_amount"] / inflate, 2)  # true PO amount

        # Corrected invoice (for negotiation, injected after send_email)
        corr_total    = approved
        corr_subtotal = round(corr_total / (1 + inv["tax_amount"] / inv["subtotal"]), 2)
        corr_tax      = round(corr_total - corr_subtotal, 2)
        corr_inv      = dict(inv, subtotal=corr_subtotal, tax_amount=corr_tax, total_amount=corr_total)

        self.vendor         = vendor
        self.inv            = inv
        self.ground_truth   = corr_inv if task_name == "expert_negotiation" else inv
        self.expected_flags = [] if task_name == "easy" else (["price_mismatch"] if task_name == "expert_negotiation" else [])
        self.erp_approved   = approved

        self.emails = {
            "email_001": _build_email_body(vendor, inv),
        }
        self.corrected_email_body = _build_email_body(vendor, corr_inv)
        self.erp_db = {vendor["name"]: {"po_number": f"PO-2025-{random.randint(100,999)}",
                                         "approved_amount": approved, "iban": vendor["iban"]}}

        # State
        self.email_content  = None
        self.erp_response   = None
        self.erp_queried    = False
        self.negotiated     = False
        self.extracted      = {}
        self.flags          = []
        self.inbox          = [{"id": "email_001", "sender": f"billing@{vendor['domain']}",
                                 "subject": "Invoice attached"}]

    def step(self, action):
        atype  = action.get("action_type", "")
        reward = 0.0
        msg    = f"Unknown action: {atype}"
        done   = False
        score  = None

        if atype == "read_email":
            eid = action.get("email_id", "")
            if eid in self.emails:
                self.email_content = self.emails[eid]
                reward = 0.05
                msg = f"Email {eid} opened."
            else:
                reward = -0.02
                msg = f"Email {eid} not found."

        elif atype == "query_erp":
            payload = action.get("api_payload", {})
            vendor_name = payload.get("vendor_name", payload.get("vendor_tax_id", ""))
            if vendor_name in self.erp_db:
                self.erp_response = self.erp_db[vendor_name]
                self.erp_queried  = True
                reward = 0.10
                msg = f"ERP: {json.dumps(self.erp_response)}"
            else:
                self.erp_response = {"error": "Vendor not found"}
                reward = -0.02
                msg = "ERP: vendor not found"

        elif atype == "extract":
            fname = action.get("field_name", "")
            fval  = action.get("field_value")
            if fname and fval is not None and self.email_content:
                self.extracted[fname] = fval
                gt = self.ground_truth
                if fname in gt:
                    try:
                        if abs(float(fval) - float(gt[fname])) < 0.02:
                            reward = 0.07
                            msg = f"Correct extraction of {fname}"
                        else:
                            reward = -0.02
                            msg = f"Wrong value for {fname}"
                    except (ValueError, TypeError):
                        if str(fval).strip() == str(gt[fname]).strip():
                            reward = 0.07
                            msg = f"Correct extraction of {fname}"
                        else:
                            reward = -0.02
                            msg = f"Wrong value for {fname}"
                else:
                    msg = f"Extracted {fname}"
            elif not self.email_content:
                reward = -0.05
                msg = "Read the email first."

        elif atype == "flag":
            fname = action.get("field_name", "")
            if fname and fname not in self.flags:
                self.flags.append(fname)
                if fname in self.expected_flags:
                    reward = 0.12
                    msg = f"Correct flag: {fname}"
                else:
                    reward = -0.05
                    msg = f"Incorrect flag: {fname}"

        elif atype == "send_email":
            target = action.get("email_id", "")
            vendor_email = f"billing@{self.vendor['domain']}"
            if self.task_name == "expert_negotiation" and not self.negotiated and target == vendor_email:
                self.negotiated = True
                self.emails["email_002"] = self.corrected_email_body
                self.inbox.append({"id": "email_002", "sender": vendor_email,
                                   "subject": "Corrected Invoice"})
                reward = 0.20
                msg = "Vendor email sent. Corrected invoice arrived (email_002)."
            else:
                reward = -0.05
                msg = f"Email sent to {target}, no reply."

        elif atype == "match_duplicate":
            reward = -0.05
            msg = "No duplicate found."

        elif atype in ("approve", "reject"):
            score = _grade(self.task_name, self.extracted, self.flags, atype,
                           self.erp_queried, self.negotiated,
                           self.ground_truth, self.expected_flags)
            reward = score
            done   = True
            msg    = f"Episode done. Score: {score}"

        return reward, done, score, msg, self.inbox, self.email_content, self.erp_response


# ── 4. Inference helper ───────────────────────────────────────────────────────
def _parse(text):
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def _infer(messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = 80,
            temperature    = 0.1,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
            use_cache      = True,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── 5. Episode runner ─────────────────────────────────────────────────────────
def run_episode(task_name, max_steps=20):
    env = MiniAPEnv(task_name)

    inbox_str = "\n".join(f"  [{e['id']}] From: {e['sender']} | Subject: {e['subject']}"
                           for e in env.inbox)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content":
         f"Task: {task_name}\n\nYour inbox:\n{inbox_str}\n\nBegin processing."},
    ]

    final_score = 0.01
    for step in range(max_steps):
        raw    = _infer(messages)
        action = _parse(raw)

        if action is None:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                              "content": "Invalid JSON. Reply with a single JSON action only."})
            continue

        reward, done, score, msg, inbox, email_c, erp_r = env.step(action)

        if done:
            final_score = score
            print(f"    step {step+1:2d}  {action.get('action_type'):20s}  r={reward:.2f}  DONE  score={score:.2f}")
            break

        inbox_str = "\n".join(f"  [{e['id']}] From: {e['sender']} | Subject: {e['subject']}"
                               for e in inbox)
        feedback = (
            f"Result: {msg}\n"
            f"Inbox: {inbox_str}\n"
            f"Email content: {email_c or '(not read yet)'}\n"
            f"ERP response: {json.dumps(erp_r) if erp_r else '(not queried)'}\n"
            f"Extracted fields: {json.dumps(env.extracted)}\n"
            f"Flags raised: {env.flags}\n"
            f"Step: {step+1}/{max_steps}\n\nWhat is your next action?"
        )
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",      "content": feedback})
        if len(messages) > 18:
            messages = messages[:2] + messages[-16:]

        print(f"    step {step+1:2d}  {action.get('action_type'):20s}  r={reward:+.2f}  msg={msg[:50]}")

    else:
        # Time-out: partial credit based on extraction progress
        n_correct = sum(1 for f in REQUIRED if f in env.extracted)
        erp_floor = 1.0 if env.erp_queried else 0.1
        final_score = round(erp_floor * 0.4 * (n_correct / len(REQUIRED)), 2)
        print(f"    TIMEOUT — partial score={final_score:.2f}")

    return final_score


# ── 6. Run evaluations ────────────────────────────────────────────────────────
N = 5
print(f"\n{'='*60}")
print(f"  Evaluating trained model  ({N} episodes per task)")
print(f"{'='*60}")

print(f"\n[EASY — {N} episodes]")
easy_scores = []
for i in range(N):
    print(f"  Episode {i+1}/{N}:")
    s = run_episode("easy")
    easy_scores.append(s)

print(f"\n[EXPERT NEGOTIATION — {N} episodes]")
neg_scores = []
for i in range(N):
    print(f"  Episode {i+1}/{N}:")
    s = run_episode("expert_negotiation")
    neg_scores.append(s)

avg_easy = round(sum(easy_scores) / N, 2)
avg_neg  = round(sum(neg_scores)  / N, 2)

print(f"\n{'='*60}")
print(f"  RESULTS SUMMARY")
print(f"{'='*60}")
print(f"  Easy               — Before: 0.26  |  After: {avg_easy:.2f}  |  Δ {avg_easy-0.26:+.2f}")
print(f"  Expert Negotiation — Before: 0.18  |  After: {avg_neg:.2f}  |  Δ {avg_neg-0.18:+.2f}")
print(f"{'='*60}")

# ── 7. Before vs After bar chart ──────────────────────────────────────────────
BEFORE = [0.26, 0.18]
AFTER  = [avg_easy, avg_neg]
LABELS = ["Easy Task", "Expert Negotiation"]

x     = np.arange(len(LABELS))
w     = 0.32
BG, PANEL, GRID, TEXT, MUTED = "#11111b", "#1e1e2e", "#313244", "#cdd6f4", "#6c7086"

fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
ax.set_facecolor(PANEL)
for sp in ax.spines.values():
    sp.set_edgecolor(GRID)
ax.tick_params(colors=MUTED, labelsize=11)
ax.grid(True, alpha=0.15, color=GRID, axis="y")

bars_b = ax.bar(x - w/2, BEFORE, w, label="Before Training  (base Llama-3.1-8B)",
                color="#f38ba8", edgecolor=BG, linewidth=0.8, zorder=3)
bars_a = ax.bar(x + w/2, AFTER,  w, label="After Training  (Unsloth GRPO fine-tuned)",
                color="#a6e3a1", edgecolor=BG, linewidth=0.8, zorder=3)

ax.axhline(0.70, color="#f9e2af", linestyle="--", alpha=0.75, linewidth=1.5,
           label="Pass threshold  (0.70)", zorder=2)
ax.set_ylim(0, 1.25)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, color=TEXT, fontsize=12, fontweight="bold")
ax.set_ylabel("Average Score  (0 – 1.0)", color=MUTED, fontsize=10)
ax.set_title(
    "Agentic Workflow Performance: Base vs. RL-Trained\n"
    "meta-llama/Llama-3.1-8B-Instruct  ·  Unsloth GRPO  ·  5 episodes per task",
    color=TEXT, fontsize=12, fontweight="bold", pad=14
)
ax.legend(fontsize=9, facecolor="#181825", labelcolor=TEXT, edgecolor=GRID, loc="upper right")

# Value labels on bars
for bar, v in zip(bars_b, BEFORE):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.025, f"{v:.2f}",
            ha="center", va="bottom", fontsize=11, color="#f38ba8", fontweight="bold")
for bar, v in zip(bars_a, AFTER):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.025, f"{v:.2f}",
            ha="center", va="bottom", fontsize=11, color="#a6e3a1", fontweight="bold")

# Delta annotations above each group
for i, (bv, av) in enumerate(zip(BEFORE, AFTER)):
    delta = av - bv
    sign  = "+" if delta >= 0 else ""
    color = "#a6e3a1" if delta >= 0 else "#f38ba8"
    ax.text(x[i], max(bv, av) + 0.09, f"{sign}{delta:.2f}",
            ha="center", va="bottom", fontsize=13, color=color, fontweight="bold")
    ax.annotate("", xy=(x[i], max(bv, av) + 0.07), xytext=(x[i], max(bv, av) + 0.03),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

plt.tight_layout()
out = "before_after_results.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"\n✓ Saved: {out}")

from IPython.display import Image, display
display(Image(out))
