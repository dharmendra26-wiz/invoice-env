"""
demo.py -- Gradio demo for the Enterprise AP-Env.

Run:   python app/demo.py
Open:  http://localhost:7860
"""

import sys, os, json, random, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from app.environment import InvoiceEnvironment
from app.models import Action


# -- Email parser --------------------------------------------------------------
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
                try: val = float(val.replace(",", ""))
                except ValueError: pass
            parsed[field] = val
    return parsed

def _parse_tax_id(body):
    m = re.search(r"Tax ID:\s*(\S+)", body, re.IGNORECASE)
    return m.group(1).strip() if m else ""


# -- Rule-based agent ----------------------------------------------------------
def _agent_action(task_name, obs, noise=0.0):
    extracted = obs.get("extracted_fields", {})
    flags     = obs.get("flags", [])
    inbox     = obs.get("inbox_status", [])
    erp       = obs.get("erp_response")
    email     = obs.get("email_content") or ""
    make_err  = random.random() < noise

    if not email:
        if inbox:
            return {"action_type": "read_email", "email_id": inbox[0]["id"]}
        return {"action_type": "reject"}

    if task_name == "expert_negotiation":
        if len(inbox) <= 1:
            if make_err: return {"action_type": "reject"}
            sender = inbox[0]["sender"] if inbox else "vendor@vendor.com"
            return {"action_type": "send_email", "email_id": sender,
                    "email_subject": "Price Discrepancy",
                    "email_body": "Please send a corrected invoice."}
        if "CORRECTED" not in email:
            return {"action_type": "read_email", "email_id": inbox[1]["id"]}

    if not erp or "error" in (erp or {}):
        if task_name == "hard":
            if make_err:
                vendor = _parse_email(email).get("vendor_name", "Unknown")
                return {"action_type": "query_erp", "api_endpoint": "/api/v1/po",
                        "api_payload": {"vendor_name": vendor}}
            tax_id = _parse_tax_id(email)
            return {"action_type": "query_erp", "api_endpoint": "/api/v2/po",
                    "api_payload": {"vendor_tax_id": tax_id}}
        else:
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


# -- Episode runner ------------------------------------------------------------
def run_episode(task_name):
    env = InvoiceEnvironment(task_name)
    obs = env.reset().model_dump()
    steps, done, step, cumulative = [], False, 0, 0.0

    while not done and step < 30:
        action = _agent_action(task_name, obs, noise=0.0)
        result = env.step(Action(**action)).model_dump()
        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]
        cumulative += reward
        step += 1
        fs = result.get("info", {}).get("final_score") if done else None
        steps.append({
            "step": step, "action": action, "reward": reward,
            "cumulative": round(cumulative, 3), "done": done,
            "message": obs["message"], "obs": obs, "final_score": fs,
        })
    return steps


# -- Rendering helpers ---------------------------------------------------------
def _action_label(action):
    base = action["action_type"]
    if action.get("field_name"):  base += f"  >  {action['field_name']}"
    if action.get("field_value"): base += f" = {action['field_value']}"
    if action.get("email_id"):    base += f"  >  {action['email_id']}"
    return base

def _reward_color(r):
    if r > 0.08:  return "#2e7d32"   # green
    if r > 0:     return "#558b2f"   # light green
    if r == 0:    return "#757575"   # grey
    return "#c62828"                 # red

def _score_badge(score):
    if score is None: return ""
    color = "#2e7d32" if score >= 0.7 else "#f57f17" if score >= 0.4 else "#c62828"
    label = "PASS" if score >= 0.7 else "PARTIAL" if score >= 0.4 else "FAIL"
    return (f"<div style='text-align:center;margin-top:16px;'>"
            f"<span style='background:{color};color:#ffffff;padding:10px 28px;"
            f"border-radius:6px;font-size:1.3em;font-weight:700;letter-spacing:0.5px;'>"
            f"Score: {score:.2f}  -  {label}</span></div>")


# -- Main demo function --------------------------------------------------------
def run_demo(task_name):
    steps = run_episode(task_name)
    log_lines, inbox_html, email_html = [], "", ""
    erp_html, fields_html, flags_html, score_html = "", "", "", ""

    for s in steps:
        obs, action, r = s["obs"], s["action"], s["reward"]
        color = _reward_color(r)
        r_sign = f"+{r:.2f}" if r >= 0 else f"{r:.2f}"

        log_lines.append(
            f"<div style='margin:4px 0;padding:9px 14px;"
            f"background:#ffffff;"
            f"border-radius:5px;border-left:4px solid {color};"
            f"font-family:monospace;font-size:0.90em;border:1px solid #e0e0e0;'>"
            f"<span style='color:#9e9e9e;font-size:0.84em;'>Step {s['step']:02d}</span>&nbsp;&nbsp;"
            f"<span style='color:#212121;font-weight:600;'>{_action_label(action)}</span>"
            f"<span style='float:right;color:{color};font-weight:700;font-size:1em;'>"
            f"{r_sign}</span><br>"
            f"<span style='color:#616161;font-size:0.87em;'>"
            f"{obs['message']}</span></div>"
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
                f"<tr style='color:#757575;font-weight:600;'><th style='text-align:left;padding:4px 10px;'>ID</th>"
                f"<th style='text-align:left;padding:4px 10px;'>From</th>"
                f"<th style='text-align:left;padding:4px 10px;'>Subject</th></tr>{rows}</table>"
            )

        if obs.get("email_content"):
            email_html = (
                f"<pre style='background:#f5f5f5;color:#212121;padding:12px;"
                f"border-radius:5px;font-size:0.82em;white-space:pre-wrap;"
                f"max-height:200px;overflow-y:auto;border:1px solid #e0e0e0;'>{obs['email_content']}</pre>"
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


# -- Gradio UI -----------------------------------------------------------------
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
    "easy":               "Easy - Read email, query ERP, extract fields, and approve.",
    "medium":             "Medium - Detect a line-item price mismatch and reject.",
    "hard":               "Hard - Handle schema drift and detect a duplicate invoice.",
    "expert_negotiation": "Expert Negotiation - Email the vendor, get a corrected invoice, then approve.",
    "expert_fraud":       "Expert Fraud - Detect a lookalike domain phishing invoice and reject.",
}

def build_demo():
    with gr.Blocks(title="Enterprise AP-Env Demo") as demo:

        gr.HTML(
            "<div style='text-align:center;padding:20px 0 8px;'>"
            "<h1 style='font-size:2em;color:#212121;font-weight:700;margin-bottom:4px;'>"
            "Enterprise AP-Env</h1>"
            "<p style='color:#616161;font-size:1em;font-weight:400;'>"
            "Multi-App AI Agent Environment</p></div>"
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
        server_port=8080,
        share=False,
        css=CSS,
        theme=gr.themes.Default(font=["Inter", "sans-serif"], primary_hue="blue", neutral_hue="gray"),
    )
