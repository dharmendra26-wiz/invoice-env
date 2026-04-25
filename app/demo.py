"""
demo.py -- Gradio demo for the Enterprise AP-Env.
Uses a real LLM agent (Llama-3.1-8B-Instruct) when HF_TOKEN is set,
falls back to rule-based agent otherwise.

Run:   python app/demo.py
Open:  http://localhost:7861
"""

import sys, os, json, re, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env from project root (safe: .env is in .gitignore)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_env_path)
    print(f"[demo] Loaded .env from {_env_path}")
except ImportError:
    print("[demo] python-dotenv not installed — skipping .env load")

import requests
import gradio as gr
from app.environment import EnterpriseAPEnvironment
from app.models import Action

# ── LLM config ────────────────────────────────────────────────────────────────
HF_TOKEN     = os.getenv("HF_TOKEN", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
LLM_ENABLED  = bool(HF_TOKEN and HF_TOKEN != "dummy-key")

SYSTEM_PROMPT = """You are an enterprise Accounts Payable AI agent processing invoices.
You interact with a multi-app environment step by step.

WORKFLOW:
1. Read emails from your inbox to find the invoice.
2. Query the ERP system to fetch the matching Purchase Order (PO) using action query_erp.
   - If the ERP returns a SCHEMA DRIFT error, retry with the field it asks for (e.g. vendor_tax_id).
3. Extract all invoice fields from the email content one by one using action extract.
4. After extracting ALL 7 fields, check for issues:
   - price_mismatch  -> invoice total does NOT match PO approved_amount: use flag action
   - tax_mismatch    -> claimed tax rate is wrong: use flag action
   - duplicate_invoice -> invoice already processed: use match_duplicate action first, then flag
   - fraud           -> sender email domain is a lookalike (e.g. "techsuppIies.com" not "techsupplies.com"): use flag action
5. For expert_negotiation: if invoice total > PO approved_amount, send email to vendor.
6. Make EXACTLY ONE final decision: approve or reject. Never use any other action types.

CRITICAL RULES:
- The ONLY valid action_type values are: read_email, query_erp, extract, flag, match_duplicate, send_email, approve, reject.
- Do NOT invent actions like compare, unflag, check_duplicate, or anything else.
- If the invoice matches the PO and no flags apply, output {"action_type": "approve"} immediately.
- If any flag was raised, output {"action_type": "reject"} immediately.
- Never repeat an action you already took.

ACTIONS:

Read an email:
  {"action_type": "read_email", "email_id": "<id from inbox>"}

Query ERP:
  {"action_type": "query_erp", "api_endpoint": "/api/v1/po", "api_payload": {"vendor_name": "Acme Corp"}}
  {"action_type": "query_erp", "api_endpoint": "/api/v2/po", "api_payload": {"vendor_tax_id": "XX-1234-56"}}

Extract a field (one at a time):
  {"action_type": "extract", "field_name": "vendor_name", "field_value": "Acme Corp"}

Flag an issue:
  {"action_type": "flag", "field_name": "price_mismatch"}

Check for duplicate invoice:
  {"action_type": "match_duplicate"}

Send email to vendor (expert_negotiation only):
  {"action_type": "send_email", "email_id": "vendor@domain.com", "email_subject": "Mismatch", "email_body": "Please send corrected invoice."}

Final decision (REQUIRED - ends the episode):
  {"action_type": "approve"}
  {"action_type": "reject"}

Fields to extract: vendor_name, invoice_number, invoice_date, due_date, subtotal, tax_amount, total_amount
Respond with ONE JSON action at a time. No markdown fences. No explanation."""


# ── LLM call with backoff ──────────────────────────────────────────────────────
def _llm_call(messages: list, retries: int = 4) -> tuple[str, list[str]]:
    """Returns (action_json_str, list_of_status_notes)."""
    wait = 15
    notes = []
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {HF_TOKEN}",
                         "Content-Type": "application/json"},
                json={"model": MODEL_NAME, "messages": messages,
                      "max_tokens": 300, "temperature": 0.0},
                timeout=60,
            )
            if resp.status_code in (402, 429) and attempt < retries:
                note = f"Rate limit ({resp.status_code}) — waiting {wait}s..."
                notes.append(note)
                time.sleep(wait)
                wait = min(wait * 2, 120)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return content, notes
        except Exception as e:
            if attempt < retries:
                notes.append(f"API error: {e} — retrying...")
                time.sleep(wait)
                wait = min(wait * 2, 60)
            else:
                raise
    raise RuntimeError("LLM call failed after all retries")


def _parse_action(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError(f"No JSON in LLM output: {text!r}")


# ── Rule-based fallback ────────────────────────────────────────────────────────
def _parse_email(body):
    parsed = {}
    patterns = {
        "vendor_name":    r"Vendor:\s*(.+)",
        "invoice_number": r"Invoice Number:\s*(\S+)",
        "invoice_date":   r"Invoice Date:\s*(\S+)",
        "due_date":       r"Due Date:\s*(\S+)",
        "subtotal":       r"Subtotal:\s*\$?([\d,]+\.?\d*)",
        "tax_amount":     r"Tax \([\d]+%\):\s*\$?([\d,]+\.?\d*)",
        "total_amount":   r"Total:\s*\$?([\d,]+\.?\d*)",
    }
    for field, pat in patterns.items():
        m = re.search(pat, body, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if field in ("subtotal", "tax_amount", "total_amount"):
                try:
                    val = float(val.replace(",", ""))
                except ValueError:
                    pass
            parsed[field] = val
    return parsed


def _parse_tax_id(body):
    m = re.search(r"Tax ID:\s*(\S+)", body, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _rule_action(task_name, obs):
    extracted = obs.get("extracted_fields", {})
    flags     = obs.get("flags", [])
    inbox     = obs.get("inbox_status", [])
    erp       = obs.get("erp_response")
    email     = obs.get("email_content") or ""

    if not email:
        if inbox:
            return {"action_type": "read_email", "email_id": inbox[0]["id"]}
        return {"action_type": "reject"}

    if task_name == "expert_negotiation":
        if len(inbox) <= 1:
            sender = inbox[0]["sender"] if inbox else "vendor@vendor.com"
            return {"action_type": "send_email", "email_id": sender,
                    "email_subject": "Price Discrepancy",
                    "email_body": "Please send a corrected invoice."}
        if "CORRECTED" not in email:
            return {"action_type": "read_email", "email_id": inbox[1]["id"]}

    if not erp or "error" in (erp or {}):
        if task_name == "hard":
            tax_id = _parse_tax_id(email)
            return {"action_type": "query_erp", "api_endpoint": "/api/v2/po",
                    "api_payload": {"vendor_tax_id": tax_id}}
        vendor = _parse_email(email).get("vendor_name", "Unknown")
        return {"action_type": "query_erp", "api_endpoint": "/api/v1/po",
                "api_payload": {"vendor_name": vendor}}

    parsed = _parse_email(email)
    fields = ["vendor_name", "invoice_number", "invoice_date",
              "due_date", "subtotal", "tax_amount", "total_amount"]
    for f in fields:
        if f not in extracted:
            val = parsed.get(f, "")
            if val:
                return {"action_type": "extract", "field_name": f, "field_value": val}

    if task_name == "easy":
        return {"action_type": "approve"}
    elif task_name == "medium":
        if "price_mismatch" not in flags:
            return {"action_type": "flag", "field_name": "price_mismatch"}
        return {"action_type": "reject"}
    elif task_name == "hard":
        if "duplicate_invoice" not in flags:
            return {"action_type": "match_duplicate"}
        if "tax_mismatch" not in flags:
            return {"action_type": "flag", "field_name": "tax_mismatch"}
        return {"action_type": "reject"}
    elif task_name == "expert_negotiation":
        return {"action_type": "approve"}
    elif task_name == "expert_fraud":
        if "fraud" not in flags:
            return {"action_type": "flag", "field_name": "fraud"}
        return {"action_type": "reject"}
    return {"action_type": "reject"}


# ── Episode runner (LLM or rule-based) ────────────────────────────────────────
def run_episode(task_name: str) -> list[dict]:
    env = EnterpriseAPEnvironment(task_name)
    obs = env.reset().model_dump()
    steps, done, step, cumulative = [], False, 0, 0.0

    # Build initial LLM messages
    inbox = obs.get("inbox_status", [])
    inbox_str = "\n".join(
        f"  [{e['id']}] From: {e['sender']} | Subject: {e['subject']}"
        for e in inbox
    ) or "  (empty)"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"Task: {task_name}\n\nYour inbox:\n{inbox_str}\n\nBegin processing."},
    ]

    while not done and step < 25:
        api_notes = []
        action = None

        if LLM_ENABLED:
            try:
                raw, api_notes = _llm_call(messages)
                action = _parse_action(raw)
            except Exception as e:
                api_notes.append(f"LLM error: {e} — using rule-based fallback")
                action = _rule_action(task_name, obs)
                raw = json.dumps(action)
        else:
            action = _rule_action(task_name, obs)
            raw = json.dumps(action)

        try:
            result = env.step(Action(**action)).model_dump()
        except Exception as e:
            steps.append({
                "step": step + 1, "action": action, "reward": 0.0,
                "cumulative": round(cumulative, 3), "done": False,
                "message": f"Invalid action: {e}", "obs": obs,
                "final_score": None, "api_notes": api_notes,
            })
            step += 1
            continue

        obs        = result["observation"]
        reward     = result["reward"]
        done       = result["done"]
        cumulative += reward
        step      += 1
        fs = result.get("info", {}).get("final_score") if done else None

        steps.append({
            "step": step, "action": action, "reward": reward,
            "cumulative": round(cumulative, 3), "done": done,
            "message": obs["message"], "obs": obs,
            "final_score": fs, "api_notes": api_notes,
        })

        # Update LLM context
        if LLM_ENABLED:
            inbox_now = obs.get("inbox_status", [])
            inbox_str = "\n".join(
                f"  [{e['id']}] From: {e['sender']} | Subject: {e['subject']}"
                for e in inbox_now
            ) or "  (empty)"
            feedback = (
                f"Result: {obs['message']}\n"
                f"Inbox: {inbox_str}\n"
                f"Email content: {obs.get('email_content') or '(not read yet)'}\n"
                f"ERP response: {json.dumps(obs.get('erp_response')) if obs.get('erp_response') else '(not queried)'}\n"
                f"Extracted fields: {json.dumps(obs.get('extracted_fields', {}))}\n"
                f"Flags raised: {obs.get('flags', [])}\n"
                f"Step: {obs.get('current_step')}/25\n"
            )
            feedback += "\nEpisode complete." if done else "\nWhat is your next action?"
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",      "content": feedback})

    return steps


# ── Rendering helpers ──────────────────────────────────────────────────────────
def _action_label(action):
    base = action["action_type"]
    if action.get("field_name"):  base += f"  >  {action['field_name']}"
    if action.get("field_value"): base += f" = {action['field_value']}"
    if action.get("email_id"):    base += f"  >  {action['email_id']}"
    return base


def _reward_color(r):
    if r > 0.08:  return "#2e7d32"
    if r > 0:     return "#558b2f"
    if r == 0:    return "#757575"
    return "#c62828"


def _score_badge(score):
    if score is None: return ""
    color = "#2e7d32" if score >= 0.7 else "#f57f17" if score >= 0.4 else "#c62828"
    label = "PASS" if score >= 0.7 else "PARTIAL" if score >= 0.4 else "FAIL"
    return (
        f"<div style='text-align:center;margin-top:16px;'>"
        f"<span style='background:{color};color:#ffffff;padding:10px 28px;"
        f"border-radius:6px;font-size:1.3em;font-weight:700;letter-spacing:0.5px;'>"
        f"Score: {score:.2f}  -  {label}</span></div>"
    )


# ── Main demo function ─────────────────────────────────────────────────────────
def run_demo(task_name):
    steps = run_episode(task_name)
    log_lines = []
    inbox_html = email_html = erp_html = fields_html = flags_html = score_html = ""

    for s in steps:
        obs, action, r = s["obs"], s["action"], s["reward"]
        color = _reward_color(r)
        r_sign = f"+{r:.2f}" if r >= 0 else f"{r:.2f}"

        # API wait notes shown as inline annotations
        notes_html = ""
        for note in s.get("api_notes", []):
            notes_html += (
                f"<div style='color:#f57f17;font-size:0.80em;font-style:italic;"
                f"padding:2px 0;'>{note}</div>"
            )

        log_lines.append(
            f"<div style='margin:4px 0;padding:9px 14px;"
            f"background:#ffffff;"
            f"border-radius:5px;border-left:4px solid {color};"
            f"font-family:monospace;font-size:0.90em;border:1px solid #e0e0e0;'>"
            f"<span style='color:#9e9e9e;font-size:0.84em;'>Step {s['step']:02d}</span>&nbsp;&nbsp;"
            f"<span style='color:#212121;font-weight:600;'>{_action_label(action)}</span>"
            f"<span style='float:right;color:{color};font-weight:700;font-size:1em;'>"
            f"{r_sign}</span><br>"
            f"<span style='color:#616161;font-size:0.87em;'>{obs['message']}</span>"
            f"{notes_html}</div>"
        )

        inbox = obs.get("inbox_status", [])
        if inbox:
            rows = "".join(
                f"<tr><td style='padding:4px 10px;color:#1565c0;font-weight:600;'>{e['id']}</td>"
                f"<td style='padding:4px 10px;color:#424242;'>{e['sender']}</td>"
                f"<td style='padding:4px 10px;color:#424242;'>{e['subject']}</td></tr>"
                for e in inbox
            )
            inbox_html = (
                f"<table style='width:100%;border-collapse:collapse;font-size:0.85em;'>"
                f"<tr style='color:#757575;font-weight:600;'>"
                f"<th style='text-align:left;padding:4px 10px;'>ID</th>"
                f"<th style='text-align:left;padding:4px 10px;'>From</th>"
                f"<th style='text-align:left;padding:4px 10px;'>Subject</th></tr>{rows}</table>"
            )

        if obs.get("email_content"):
            email_html = (
                f"<pre style='background:#f5f5f5;color:#212121;padding:12px;"
                f"border-radius:5px;font-size:0.82em;white-space:pre-wrap;"
                f"max-height:200px;overflow-y:auto;border:1px solid #e0e0e0;'>"
                f"{obs['email_content']}</pre>"
            )

        if obs.get("erp_response"):
            ej = json.dumps(obs["erp_response"], indent=2)
            ec = "#c62828" if "error" in obs["erp_response"] else "#2e7d32"
            erp_html = (
                f"<pre style='background:#f5f5f5;color:{ec};padding:12px;"
                f"border-radius:5px;font-size:0.82em;white-space:pre-wrap;"
                f"max-height:200px;overflow-y:auto;border:1px solid #e0e0e0;'>{ej}</pre>"
            )

        ef = obs.get("extracted_fields", {})
        if ef:
            rows = "".join(
                f"<tr><td style='padding:3px 10px;color:#1565c0;font-weight:600;'>{k}</td>"
                f"<td style='padding:3px 10px;color:#424242;'>{v}</td></tr>"
                for k, v in ef.items()
            )
            fields_html = (
                f"<table style='width:100%;border-collapse:collapse;font-size:0.85em;'>"
                f"<tr style='color:#757575;font-weight:600;'>"
                f"<th style='text-align:left;padding:3px 10px;'>Field</th>"
                f"<th style='text-align:left;padding:3px 10px;'>Value</th></tr>{rows}</table>"
            )

        fl = obs.get("flags", [])
        if fl:
            badges = " ".join(
                f"<span style='background:#c62828;color:#ffffff;padding:3px 10px;"
                f"border-radius:4px;font-size:0.82em;margin:2px;font-weight:600;'>"
                f"FLAG: {f}</span>" for f in fl
            )
            flags_html = f"<div style='padding:8px;'>{badges}</div>"
        else:
            flags_html = "<span style='color:#9e9e9e;font-size:0.85em;'>No flags raised yet.</span>"

        if s["final_score"] is not None:
            score_html = _score_badge(s["final_score"])

    log_html = (
        "<div style='background:#fafafa;border:1px solid #e0e0e0;border-radius:8px;"
        "padding:12px;max-height:420px;overflow-y:auto;'>" + "\n".join(log_lines) + "</div>"
    )
    return log_html, inbox_html, email_html, erp_html, fields_html, flags_html, score_html


# ── Gradio UI ──────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body, .gradio-container {
  background: #f5f5f5 !important;
  color: #212121 !important;
  font-family: 'Inter', sans-serif !important;
}
button { font-family: 'Inter', sans-serif !important; }
label { color: #424242 !important; font-weight: 500 !important; }
"""

TASK_DESCRIPTIONS = {
    "easy":               "Easy - Read email, query ERP, extract 7 fields, and approve.",
    "medium":             "Medium - Detect a line-item price mismatch and reject.",
    "hard":               "Hard - Handle ERP schema drift and detect a duplicate invoice.",
    "expert_negotiation": "Expert Negotiation - Email the vendor, get a corrected invoice, then approve.",
    "expert_fraud":       "Expert Fraud - Detect a lookalike domain phishing invoice and reject.",
}


def build_demo():
    # Model badge shown in header
    if LLM_ENABLED:
        agent_badge = (
            f"<span style='background:#1565c0;color:#fff;padding:3px 10px;"
            f"border-radius:4px;font-size:0.78em;font-weight:600;margin-left:10px;'>"
            f"LLM: {MODEL_NAME.split('/')[-1]}</span>"
        )
    else:
        agent_badge = (
            "<span style='background:#f57f17;color:#fff;padding:3px 10px;"
            "border-radius:4px;font-size:0.78em;font-weight:600;margin-left:10px;'>"
            "Rule-Based Agent (set HF_TOKEN to enable LLM)</span>"
        )

    with gr.Blocks(title="Enterprise AP-Env Demo", css=CSS) as demo:

        gr.HTML(
            f"<div style='text-align:center;padding:20px 0 8px;'>"
            f"<h1 style='font-size:2em;color:#212121;font-weight:700;margin-bottom:4px;'>"
            f"Enterprise AP Environment</h1>"
            f"<p style='color:#616161;font-size:1em;font-weight:400;'>"
            f"Multi-App AI Agent Environment&nbsp;{agent_badge}</p></div>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                task_dd = gr.Dropdown(
                    choices=list(TASK_DESCRIPTIONS.keys()),
                    value="easy", label="Select Task",
                )
                task_desc = gr.HTML(
                    f"<div style='color:#424242;font-size:0.88em;padding:6px 0;'>"
                    f"{TASK_DESCRIPTIONS['easy']}</div>"
                )
                run_btn = gr.Button("Run Episode", variant="primary", size="lg")
                score_out = gr.HTML(label="Final Score")
                gr.HTML("<hr style='border-color:#e0e0e0;margin:12px 0;'>")
                gr.HTML(
                    "<div style='background:#ffffff;border:1px solid #e0e0e0;border-radius:6px;"
                    "padding:12px;font-size:0.85em;line-height:2;'>"
                    "<b style='color:#212121;font-size:0.92em;'>Reward Breakdown</b><br>"
                    "<span style='color:#2e7d32;font-weight:600;'>+0.07</span>"
                    "<span style='color:#424242;'> Correct field extraction</span><br>"
                    "<span style='color:#2e7d32;font-weight:600;'>+0.10</span>"
                    "<span style='color:#424242;'> ERP query success</span><br>"
                    "<span style='color:#2e7d32;font-weight:600;'>+0.05</span>"
                    "<span style='color:#424242;'> Email read</span><br>"
                    "<span style='color:#2e7d32;font-weight:600;'>+0.20</span>"
                    "<span style='color:#424242;'> Negotiation success</span><br>"
                    "<span style='color:#2e7d32;font-weight:600;'>+0.12</span>"
                    "<span style='color:#424242;'> Correct flag raised</span><br>"
                    "<span style='color:#757575;font-size:0.88em;'>"
                    "Final score: 40% fields / 30% flags / 30% decision</span></div>"
                )

            with gr.Column(scale=3):
                gr.HTML("<div style='font-size:0.95em;font-weight:600;color:#212121;margin-bottom:6px;'>"
                        "Agent Step Log</div>")
                log_out = gr.HTML(
                    "<div style='background:#fafafa;border:1px solid #e0e0e0;border-radius:8px;"
                    "padding:24px;color:#9e9e9e;font-style:italic;text-align:center;'>"
                    "Select a task and click Run Episode to watch the agent work.</div>"
                )

        gr.HTML("<hr style='border-color:#e0e0e0;margin:8px 0;'>")

        def _hdr(label):
            return f"<div style='font-size:0.90em;font-weight:600;color:#424242;margin-bottom:4px;'>{label}</div>"
        def _empty(msg):
            return f"<div style='color:#9e9e9e;font-size:0.85em;padding:8px;'>{msg}</div>"

        with gr.Row():
            with gr.Column():
                gr.HTML(_hdr("Inbox"))
                inbox_out = gr.HTML(_empty("No emails yet"))
            with gr.Column():
                gr.HTML(_hdr("Email Content"))
                email_out = gr.HTML(_empty("Email not read yet"))

        with gr.Row():
            with gr.Column():
                gr.HTML(_hdr("ERP Response"))
                erp_out = gr.HTML(_empty("ERP not queried yet"))
            with gr.Column():
                gr.HTML(_hdr("Extracted Fields"))
                fields_out = gr.HTML(_empty("No fields extracted yet"))

        with gr.Row():
            with gr.Column():
                gr.HTML(_hdr("Flags Raised"))
                flags_out = gr.HTML(_empty("No flags raised"))

        def update_desc(task):
            return (f"<div style='color:#424242;font-size:0.88em;padding:6px 0;'>"
                    f"{TASK_DESCRIPTIONS.get(task, '')}</div>")

        task_dd.change(update_desc, inputs=task_dd, outputs=task_desc)
        run_btn.click(
            run_demo, inputs=task_dd,
            outputs=[log_out, inbox_out, email_out, erp_out,
                     fields_out, flags_out, score_out],
        )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
    )
