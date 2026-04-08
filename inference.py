import os
import sys
import json
import requests

API_BASE_URL=os.getenv("API_BASE_URL","https://router.huggingface.co/v1")
MODEL_NAME=os.getenv("MODEL_NAME","Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN=os.getenv("HF_TOKEN") or os.getenv("API_KEY","dummy-key")
ENV_URL=os.getenv("ENV_URL","http://localhost:7860")
BENCHMARK="invoice-env"

try:
    from openai import OpenAI
    client=OpenAI(base_url=API_BASE_URL,api_key=HF_TOKEN)
except Exception as e:
    print(f"[ERROR] OpenAI client init failed: {e}",file=sys.stderr)
    sys.exit(1)

SYSTEM_PROMPT="""You are an accounts payable agent. You process invoices by taking actions.

Available actions (respond with ONLY a JSON object):
1. Extract a field:
   {"action_type":"extract","field_name":"vendor_name","field_value":"ABC Corp"}

2. Match PO:
   {"action_type":"match_po"}

3. Flag an issue:
   {"action_type":"flag","field_name":"price_mismatch"}
   {"action_type":"flag","field_name":"duplicate_invoice"}
   {"action_type":"flag","field_name":"tax_mismatch"}

4. Check duplicate:
   {"action_type":"match_duplicate"}

5. Final decision:
   {"action_type":"approve"}
   {"action_type":"reject"}

Fields to extract: vendor_name, invoice_number, invoice_date, due_date, subtotal, tax_amount, total_amount

Respond with ONE JSON action at a time. No explanation, just the JSON."""

def run_task(task_name:str):
    try:
        resp=requests.post(f"{ENV_URL}/reset",params={"task_name":task_name},timeout=30)
        resp.raise_for_status()
        obs=resp.json()
    except Exception as e:
        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")
        print(f"[STEP] step=1 action=null reward=0.00 done=true error={e}")
        print(f"[END] success=false steps=1 score=0.00 rewards=0.00")
        return 0.0

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"Invoice:\n{obs['invoice_text']}\n\nPO Data:\n{obs['po_data']}\n\nExtract all fields, check for issues, then approve or reject."}
    ]

    step=0
    rewards=[]
    done=False

    while not done and step<20:
        try:
            completion=client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=200,
                temperature=0.0
            )
            action_str=completion.choices[0].message.content.strip()

            if "```" in action_str:
                action_str=action_str.split("```")[1]
                if action_str.startswith("json"):
                    action_str=action_str[4:]
            action_str=action_str.strip()

            action_dict=json.loads(action_str)

            step_resp=requests.post(
                f"{ENV_URL}/step",
                params={"task_name":task_name},
                json=action_dict,
                timeout=30
            )
            step_resp.raise_for_status()
            result=step_resp.json()

            reward=result["reward"]
            done=result["done"]
            message=result["observation"]["message"]

            rewards.append(reward)
            step+=1

            print(f"[STEP] step={step} action={action_str.replace(chr(10),'')} reward={reward:.2f} done={str(done).lower()} error=null")

            messages.append({"role":"assistant","content":action_str})
            messages.append({"role":"user","content":f"Result: {message}\nExtracted so far: {result['observation']['extracted_fields']}\nFlags: {result['observation']['flags']}\nWhat is your next action?"})

        except Exception as e:
            step+=1
            rewards.append(0.0)
            print(f"[STEP] step={step} action=null reward=0.00 done=false error={e}")
            break

    score=max(rewards) if rewards else 0.0
    success=score>=0.7
    rewards_str=",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}")
    return score

def main():
    tasks=["easy","medium","hard"]
    scores={}
    for task in tasks:
        score=run_task(task)
        scores[task]=score
    print("\n=== FINAL SCORES ===")
    for t,s in scores.items():
        print(f"{t}: {s:.2f}")

if __name__=="__main__":
    main()