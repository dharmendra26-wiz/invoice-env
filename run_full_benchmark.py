import subprocess
import time
import os
import sys

def main():
    print("========================================")
    print(" STARTING 50-EPISODE 70B BENCHMARK")
    print("========================================")
    
    # 1. Start the FastAPI backend
    print("[1/4] Starting FastAPI backend on port 7860...")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    time.sleep(5)  # Wait for server to boot
    
    # 2. Run inference.py
    print("\n[2/4] Running LLM Inference (This will take ~45 minutes)...")
    env = os.environ.copy()
    env["MODEL_NAME"] = "meta-llama/Llama-3.3-70B-Instruct"
    if not env.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN environment variable is not set. Export it before running the benchmark.")
    
    try:
        subprocess.run(
            [sys.executable, "inference.py", "--all", "--episodes", "50"],
            env=env,
            check=True
        )
    except subprocess.CalledProcessError:
        print("Error during inference.")
        
    # 3. Plot the results
    print("\n[3/4] Generating Reward Curves...")
    subprocess.run([sys.executable, "plot_llm_results.py", "training_results.json"])
    
    # 4. Cleanup
    print("\n[4/4] Shutting down backend server...")
    server.terminate()
    server.wait()
    
    print("\n========================================")
    print(" BENCHMARK COMPLETE! Check the PNG files.")
    print("========================================")

if __name__ == "__main__":
    main()
