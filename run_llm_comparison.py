import os
import json
import statistics
from inference import run_task, MODEL_NAME

os.environ["HF_TOKEN"] = "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"
# Use the local API server if running locally, or hit the HF space directly
os.environ["ENV_URL"] = "http://localhost:7860"

def evaluate_model(model_id: str, episodes: int = 20):
    print(f"\n==============================================")
    print(f" EVALUATING: {model_id} FOR {episodes} EPISODES")
    print(f"==============================================")
    os.environ["MODEL_NAME"] = model_id
    
    scores = []
    for i in range(episodes):
        print(f"\n--- Episode {i+1}/{episodes} ---")
        try:
            score = run_task("expert_fraud")
            scores.append(score)
            print(f"  -> Score: {score}")
        except Exception as e:
            print(f"  -> Failed: {e}")
            scores.append(0.0)
            
    avg_score = statistics.mean(scores) if scores else 0
    print(f"\n[FINAL] {model_id} Average Score: {avg_score:.2f}")
    
    # Save results
    filename = f"{model_id.split('/')[-1]}_results.json"
    with open(filename, "w") as f:
        json.dump({"model": model_id, "episodes": episodes, "scores": scores, "average": avg_score}, f, indent=2)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    # Start the backend server if not running
    import subprocess
    import time
    import requests
    
    server_process = None
    try:
        requests.get("http://localhost:7860/health")
        print("Backend server already running.")
    except:
        print("Starting local backend server on port 7860...")
        server_process = subprocess.Popen(["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"])
        time.sleep(3) # wait for boot
        
    try:
        evaluate_model("meta-llama/Llama-3.1-8B-Instruct", 20)
        evaluate_model("Qwen/Qwen2.5-72B-Instruct", 20)  # Example 70B+ model available on HF free tier
    finally:
        if server_process:
            server_process.terminate()
            print("Terminated backend server.")
