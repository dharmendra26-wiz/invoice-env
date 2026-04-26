from dotenv import load_dotenv
import os, requests

load_dotenv(".env")
token = os.getenv("HF_TOKEN", "")
print(f"Token loaded: {token[:10]}...{token[-4:]}")
print(f"LLM_ENABLED: {bool(token and token != 'dummy-key')}")

# Quick API ping
resp = requests.post(
    "https://router.huggingface.co/v1/chat/completions",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 5,
    },
    timeout=30,
)
print(f"API status: {resp.status_code}")
if resp.status_code == 200:
    print(f"LLM reply: {resp.json()['choices'][0]['message']['content']}")
else:
    print(f"Error: {resp.text[:400]}")
