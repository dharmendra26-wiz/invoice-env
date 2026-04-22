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


# -- Email parser (same as train.py) -------------------------------------------
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


# -- Rule-based agent (mirrors train.py) ---------------------------------------
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

    # expert_negotiation: negotiate BEFORE extracting
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
ACTION_ICONS = {
    "read_email": "MAIL", "query_erp": "ERP", "extract": "FIELD",
    "flag": "FLAG", "match_duplicate": "DUP", "send_email": "SEND",
    "approve": "OK", "reject": "NO",
}

def _action_label(action):
    icon = ACTION_ICONS.get(action["action_type"], ">")
    base = f"[{icon}] {action['action_type']}"
    if action.get("field_name"):  base += f" -> {action['field_name']}"
    if action.get("field_value"): base += f" = {action['field_value']}"
    if action.get("email_id"):    base += f" -> {action['email_id']}"
    return base

def _reward_color(r):
    if r > 0.08:  return "#4CAF50"
    if r > 0:     return "#8BC34A"
    if r == 0:    return "#9E9E9E"
    return "#F44336"

def _score_badge(score):
    if score is None: return ""
    color = "#4CAF50" if score >= 0.7 else "#FF9800" if score >= 0.4 else "#F44336"
    label = "PASS" if score >= 0.7 else "PARTIAL" if score >= 0.4 else "FAIL"
    return (f"<div style='text-align:center;margin-top:12px;'>"
            f"<span style='background:{color};color:white;padding:10px 28px;"
            f"border-radius:24px;font-size:1.4em;font-weight:700;'>"
            f"Final Score: {score:.2f} - {label}</span></div>")


# -- Main demo function -------------------------------------------------------
def run_demo(task_name):
    steps = run_episode(task_name)
    log_lines, inbox_html, email_html = [], "", ""
    erp_html, fields_html, flags_html, score_html = "", "", "", ""

    for s in steps:
        obs, action, r = s["obs"], s["action"], s["reward"]
        color = _reward_color(r)
        r_sign = f"+{r:.2f}" if r >= 0 else f"{r:.2f}"

        log_lines.append(
            f"<div style='margin:4px 0;padding:6px 10px;background:#1e1e2e;"
            f"border-radius:6px;border-left:3px solid {color};"
            f"font-family:monospace;font-size:0.88em;'>"
            f"<span style='color:#888;'>Step {s['step']:02d}</span>&nbsp;&nbsp;"
            f"<span style='color:#cdd6f4;'>{_action_label(action)}</span>"
            f"<span style='float:right;color:{color};font-weight:bold;'>"
            f"{r_sign}</span><br>"
            f"<span style='color:#6c7086;font-size:0.85em;'>"
            f"{obs['message']}</span></div>"
        )

        inbox = obs.get("inbox_status", [])
        if inbox:
            rows = "".join(
                f"<tr><td style='padding:4px 10px;color:#89b4fa;'>{e['id']}</td>"
                f"<td style='padding:4px 10px;'>{e['sender']}</td>"
                f"<td style='padding:4px 10px;'>{e['subject']}</td></tr>"
                for e in inbox
            )
            inbox_html = (
                f"<table style='width:100%;border-collapse:collapse;"
                f"font-size:0.85em;color:#cdd6f4;'>"
                f"<tr style='color:#6c7086;'><th>ID</th><th>From</th>"
                f"<th>Subject</th></tr>{rows}</table>"
            )

        if obs.get("email_content"):
            email_html = (
                f"<pre style='background:#1e1e2e;color:#cdd6f4;padding:12px;"
                f"border-radius:8px;font-size:0.82em;white-space:pre-wrap;"
                f"max-height:200px;overflow-y:auto;'>{obs['email_content']}</pre>"
            )

        if obs.get("erp_response"):
            ej = json.dumps(obs["erp_response"], indent=2)
            ec = "#F44336" if "error" in obs["erp_response"] else "#4CAF50"
            erp_html = (
                f"<pre style='background:#1e1e2e;color:{ec};padding:12px;"
                f"border-radius:8px;font-size:0.82em;white-space:pre-wrap;"
                f"max-height:200px;overflow-y:auto;'>{ej}</pre>"
            )

        ef = obs.get("extracted_fields", {})
        if ef:
            rows = "".join(
                f"<tr><td style='padding:3px 10px;color:#a6e3a1;'>{k}</td>"
                f"<td style='padding:3px 10px;color:#cdd6f4;'>{v}</td></tr>"
                for k, v in ef.items()
            )
            fields_html = (
                f"<table style='width:100%;border-collapse:collapse;"
                f"font-size:0.85em;'><tr style='color:#6c7086;'>"
                f"<th>Field</th><th>Value</th></tr>{rows}</table>"
            )

        fl = obs.get("flags", [])
        if fl:
            badges = " ".join(
                f"<span style='background:#F44336;color:white;padding:3px 10px;"
                f"border-radius:12px;font-size:0.82em;margin:2px;'>"
                f"FLAG: {f}</span>" for f in fl
            )
            flags_html = f"<div style='padding:8px;'>{badges}</div>"
        else:
            flags_html = ("<span style='color:#6c7086;font-size:0.85em;'>"
                          "No flags raised yet.</span>")

        if s["final_score"] is not None:
            score_html = _score_badge(s["final_score"])

    log_html = (
        "<div style='background:#181825;border-radius:10px;padding:10px;"
        "max-height:380px;overflow-y:auto;'>" + "\n".join(log_lines) + "</div>"
    )
    return log_html, inbox_html, email_html, erp_html, fields_html, flags_html, score_html


# -- Gradio UI -----------------------------------------------------------------
CSS = """
body, .gradio-container { background: #11111b !important; color: #cdd6f4 !important; }
.gr-button-primary { background: linear-gradient(135deg,#89b4fa,#cba6f7) !important;
                     border:none !important; color:#11111b !important;
                     font-weight:700 !important; border-radius:8px !important; }
.gr-button-primary:hover { opacity:0.88 !important; }
h1,h2,h3 { color: #cdd6f4 !important; }
label { color: #a6adc8 !important; }
.gr-panel { background: #1e1e2e !important; border: 1px solid #313244 !important;
            border-radius: 10px !important; }
"""

TASK_DESCRIPTIONS = {
    "easy":               "Easy -- Read email, Query ERP, Extract fields, Approve",
    "medium":             "Medium -- Detect line-item price mismatch & Reject",
    "hard":               "Hard -- Schema drift + duplicate invoice detection",
    "expert_negotiation": "Expert: Negotiation -- Email vendor, get corrected invoice",
    "expert_fraud":       "Expert: Fraud -- Lookalike domain phishing detection",
}

def build_demo():
    with gr.Blocks(title="Enterprise AP-Env Demo") as demo:
        gr.HTML(
            "<div style='text-align:center;padding:20px 0 10px;'>"
            "<h1 style='font-size:2em;background:linear-gradient(135deg,#89b4fa,#cba6f7);"
            "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
            "font-weight:800;margin-bottom:4px;'>Enterprise AP-Env</h1>"
            "<p style='color:#6c7086;font-size:1em;'>"
            "Multi-App AI Agent Environment -- Meta AI Hackathon Finals</p></div>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                task_dd = gr.Dropdown(
                    choices=list(TASK_DESCRIPTIONS.keys()),
                    value="easy", label="Select Task",
                )
                task_desc = gr.HTML(
                    f"<div style='color:#a6adc8;font-size:0.88em;padding:4px 0;'>"
                    f"{TASK_DESCRIPTIONS['easy']}</div>"
                )
                run_btn = gr.Button("Run Episode", variant="primary", size="lg")
                score_out = gr.HTML(label="Final Score")
                gr.HTML("<hr style='border-color:#313244;margin:10px 0;'>")
                gr.HTML(
                    "<div style='color:#6c7086;font-size:0.82em;line-height:1.6;'>"
                    "<b style='color:#a6adc8;'>Reward Breakdown</b><br>"
                    "Correct extraction: +0.07<br>"
                    "ERP query success: +0.10<br>"
                    "Email read: +0.05<br>"
                    "Negotiate: +0.20<br>"
                    "Correct flag: +0.12<br>"
                    "Final grader: 40/30/30%</div>"
                )

            with gr.Column(scale=3):
                gr.HTML("<b style='color:#89b4fa;'>Step-by-Step Agent Log</b>")
                log_out = gr.HTML(
                    "<div style='background:#181825;border-radius:10px;padding:20px;"
                    "color:#6c7086;font-style:italic;'>"
                    "Click Run Episode to start.</div>"
                )

        gr.HTML("<hr style='border-color:#313244;margin:8px 0;'>")

        with gr.Row():
            with gr.Column():
                gr.HTML("<b style='color:#89b4fa;'>Inbox</b>")
                inbox_out = gr.HTML("<div style='color:#6c7086;font-size:0.85em;'>--</div>")
            with gr.Column():
                gr.HTML("<b style='color:#89b4fa;'>Email Content</b>")
                email_out = gr.HTML("<div style='color:#6c7086;font-size:0.85em;'>--</div>")

        with gr.Row():
            with gr.Column():
                gr.HTML("<b style='color:#89b4fa;'>ERP Response</b>")
                erp_out = gr.HTML("<div style='color:#6c7086;font-size:0.85em;'>--</div>")
            with gr.Column():
                gr.HTML("<b style='color:#89b4fa;'>Extracted Fields</b>")
                fields_out = gr.HTML("<div style='color:#6c7086;font-size:0.85em;'>--</div>")

        with gr.Row():
            with gr.Column():
                gr.HTML("<b style='color:#89b4fa;'>Flags Raised</b>")
                flags_out = gr.HTML("<div style='color:#6c7086;font-size:0.85em;'>None</div>")

        def update_desc(task):
            return (f"<div style='color:#a6adc8;font-size:0.88em;padding:4px 0;'>"
                    f"{TASK_DESCRIPTIONS.get(task,'')}</div>")

        task_dd.change(update_desc, inputs=task_dd, outputs=task_desc)
        run_btn.click(
            run_demo, inputs=task_dd,
            outputs=[log_out, inbox_out, email_out, erp_out,
                     fields_out, flags_out, score_out],
        )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
