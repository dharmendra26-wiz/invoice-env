"""Quick headless test: run one 'easy' episode with LLM and print results."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(".env")

# Monkey-patch so we can import demo without launching Gradio
import app.demo as demo

print(f"\n{'='*55}")
print(f"  LLM_ENABLED : {demo.LLM_ENABLED}")
print(f"  MODEL       : {demo.MODEL_NAME}")
print(f"  TOKEN       : {demo.HF_TOKEN[:10]}...{demo.HF_TOKEN[-4:] if demo.HF_TOKEN else ''}")
print(f"{'='*55}\n")

if not demo.LLM_ENABLED:
    print("❌ LLM not enabled — check HF_TOKEN in .env")
    sys.exit(1)

print("Running 'easy' episode with real LLM...\n")
steps = demo.run_episode("easy")

for s in steps:
    reward_sign = f"+{s['reward']:.2f}" if s['reward'] >= 0 else f"{s['reward']:.2f}"
    agent = "LLM" if demo.LLM_ENABLED else "Rule"
    print(f"  Step {s['step']:02d} [{agent}]  {s['action']['action_type']:<18}  reward={reward_sign}  cumulative={s['cumulative']:.3f}")
    if s['api_notes']:
        for note in s['api_notes']:
            print(f"           ⚠️  {note}")
    if s['final_score'] is not None:
        label = "PASS" if s['final_score'] >= 0.7 else "PARTIAL" if s['final_score'] >= 0.4 else "FAIL"
        print(f"\n{'='*55}")
        print(f"  FINAL SCORE: {s['final_score']:.2f}  →  {label}")
        print(f"{'='*55}\n")
