"""
inference.py — LLM agent that drives the invoice-env environment.

Usage:
    python inference.py                  # run easy/medium/hard
    python inference.py --all            # run all 5 tasks
    python inference.py --task expert_fraud

Environment variables:
    ENV_URL        default: http://localhost:7860
    MODEL_NAME     default: Qwen/Qwen2.5-72B-Instruct
    HF_TOKEN       HuggingFace API token
    API_BASE_URL   default: https://router.huggingface.co/v1
"""

import os, sys, json, re, argparse, requests

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
BENCHMARK    = "invoice-env"

# ── LLM call ──────────────────────────────────────────────────────────────────
def llm_call(messages: list) -> str:
    resp = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
        json={"model": MODEL_NAME, "messages": messages,
              "max_tokens": 300, "temperature": 0.0},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an enterprise Accounts Payable AI agent processing invoices.
You interact with a multi-app environment step by step.

WORKFLOW:
1. Read emails from your inbox to find the invoice.
2. Query the ERP system to fetch the matching Purchase Order (PO).
   - If the ERP returns a SCHEMA DRIFT error, retry with the field it asks for (e.g. vendor_tax_id).
3. Extract all invoice fields from the email content.
4. Compare invoice vs PO. Flag any mismatches.
   - price_mismatch  → invoice line item prices differ from PO
   - tax_mismatch    → claimed tax rate differs from PO
   - duplicate_invoice → same invoice number was processed before
   - fraud          → email sender domain looks like a lookalike (e.g. "techsuppIies.com" vs "techsupplies.com")
5. For expert_negotiation tasks: if invoice total > PO approved_amount, send an email to the vendor.
6. Make a final decision: approve or reject.

ACTIONS — respond with ONLY a JSON object, no explanation:

Read an email:
  {"action_type": "read_email", "email_id": "<id from inbox>"}

Query ERP:
  {"action_type": "query_erp", "api_endpoint": "/api/v1/po", "api_payload": {"vendor_name": "Acme Corp"}}
  {"action_type": "query_erp", "api_endpoint": "/api/v2/po", "api_payload": {"vendor_tax_id": "XX-1234-56"}}

Extract a field:
  {"action_type": "extract", "field_name": "vendor_name", "field_value": "Acme Corp"}

Flag an issue:
  {"action_type": "flag", "field_name": "price_mismatch"}

Check duplicate:
  {"action_type": "match_duplicate"}

Send email to vendor:
  {"action_type": "send_email", "email_id": "vendor@domain.com", "email_subject": "Mismatch", "email_body": "Please clarify."}

Final decision:
  {"action_type": "approve"}
  {"action_type": "reject"}

Fields to extract: vendor_name, invoice_number, invoice_date, due_date, subtotal, tax_amount, total_amount
Respond with ONE JSON action at a time. No markdown fences."""


# ── JSON extractor (robust) ───────────────────────────────────────────────────
def parse_action(text: str) -> dict:
    # strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # grab first {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError(f"No JSON found in: {text!r}")


# ── Single-task runner ────────────────────────────────────────────────────────
def run_task(task_name: str) -> float:
    # Reset environment
    try:
        obs = requests.post(f"{ENV_URL}/reset",
                            params={"task_name": task_name}, timeout=30).json()
    except Exception as e:
        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")
        print(f"[STEP] step=1 action=null reward=0.00 done=true error={e}")
        print(f"[END] success=false steps=1 score=0.00 rewards=0.00")
        return 0.0

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    # Build initial context for LLM
    inbox = obs.get("inbox_status", [])
    inbox_str = "\n".join(
        f"  [{e['id']}] From: {e['sender']} | Subject: {e['subject']}"
        for e in inbox
    ) or "  (empty)"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content":
            f"Task: {task_name}\n\nYour inbox:\n{inbox_str}\n\nBegin processing."},
    ]

    step, rewards, done = 0, [], False

    while not done and step < 25:
        # --- get LLM action ---
        try:
            raw = llm_call(messages)
            action = parse_action(raw)
            action_str = json.dumps(action)
        except Exception as e:
            step += 1
            rewards.append(0.0)
            print(f"[STEP] step={step} action=null reward=0.00 done=false error={e}")
            # tell LLM it failed
            messages.append({"role": "assistant", "content": str(raw) if 'raw' in dir() else ""})
            messages.append({"role": "user", "content": f"Parse error: {e}. Reply with valid JSON only."})
            continue

        # --- send to environment ---
        try:
            result = requests.post(f"{ENV_URL}/step",
                                   params={"task_name": task_name},
                                   json=action, timeout=30).json()
        except Exception as e:
            step += 1
            rewards.append(0.0)
            print(f"[STEP] step={step} action={action_str} reward=0.00 done=false error={e}")
            break

        reward = result["reward"]
        done   = result["done"]
        obs    = result["observation"]
        msg    = obs["message"]
        rewards.append(reward)
        step  += 1
        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

        # Build feedback for LLM
        inbox_now = obs.get("inbox_status", [])
        inbox_str = "\n".join(
            f"  [{e['id']}] From: {e['sender']} | Subject: {e['subject']}"
            for e in inbox_now
        ) or "  (empty)"

        feedback = (
            f"Result: {msg}\n"
            f"Inbox: {inbox_str}\n"
            f"Email content: {obs.get('email_content') or '(not read yet)'}\n"
            f"ERP response: {json.dumps(obs.get('erp_response')) if obs.get('erp_response') else '(not queried)'}\n"
            f"Extracted fields: {json.dumps(obs.get('extracted_fields', {}))}\n"
            f"Flags raised: {obs.get('flags', [])}\n"
            f"Step: {obs.get('current_step')}/25\n"
        )
        if done:
            feedback += f"\nEpisode complete. Final score included in result."
        else:
            feedback += "\nWhat is your next action?"

        messages.append({"role": "assistant", "content": action_str})
        messages.append({"role": "user",      "content": feedback})

    # --- final score ---
    final_score = result.get("info", {}).get("final_score", max(rewards) if rewards else 0.01) if 'result' in dir() else 0.01
    score   = float(final_score) if final_score else (max(rewards) if rewards else 0.01)
    score   = round(min(0.99, max(0.01, score)), 2)
    success = score >= 0.7
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}")
    return score


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Invoice-env LLM inference agent")
    parser.add_argument("--task",  default=None,  help="Single task name to run")
    parser.add_argument("--all",   action="store_true", help="Run all 5 tasks")
    args = parser.parse_args()

    all_tasks = ["easy", "medium", "hard", "expert_negotiation", "expert_fraud"]
    if args.all:
        tasks = all_tasks
    elif args.task:
        tasks = [args.task]
    else:
        tasks = ["easy", "medium", "hard"]

    scores = {}
    for t in tasks:
        scores[t] = run_task(t)

    print("\n=== FINAL SCORES ===")
    for t, s in scores.items():
        status = "✓ PASS" if s >= 0.7 else "✗ FAIL"
        print(f"  {t:<22} {s:.2f}  {status}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'AVERAGE':<22} {avg:.2f}")


if __name__ == "__main__":
    main()