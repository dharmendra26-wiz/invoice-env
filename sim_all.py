import os
import json
import random

# Read train.py to dynamically patch the agent noise
with open('train.py', 'r', encoding='utf-8') as f:
    original_train_py = f.read()

def run_simulation(model_name, noise_level):
    print(f"\n===============================================")
    print(f" SIMULATING {model_name} ACROSS ALL TASKS")
    print(f"===============================================")
    
    # Patch train.py logic
    # First, replace the hardcoded expert_fraud loop with a full loop
    patched = original_train_py.replace(
        '''    for ep in range(total_episodes):
        current_task = "expert_fraud"  # HARDCODED FOR LLM EVALUATION
        
        # Run episode with global decay logic
        r = runner(current_task, ep, total_episodes)
        all_rewards[current_task].append(r)
        recent_scores.append(r)
        
        # Logging
        if len(recent_scores) > 0:
            avg = sum(recent_scores) / len(recent_scores)
            filled = int(avg * 20)
            bar = "#" * filled + "-" * (20 - filled)
            print(f"  Ep {ep+1:>3}/{total_episodes} | Task: {current_task:<18} | [{bar}] {avg:.3f}")''',
        '''    for current_task in TASKS:
        recent_scores = []
        for ep in range(episodes):
            r = runner(current_task, ep, episodes)
            all_rewards[current_task].append(r)
            recent_scores.append(r)
            
            if len(recent_scores) > 0:
                avg = sum(recent_scores) / len(recent_scores)
                filled = int(avg * 20)
                bar = "#" * filled + "-" * (20 - filled)
                print(f"  Ep {ep+1:>3}/{episodes} | Task: {current_task:<18} | [{bar}] {avg:.3f}")'''
    )
    
    # Now patch the noise
    patched = patched.replace(
        'noise    = max(0.0, 0.55 - (episode / total_episodes) * 0.65)',
        f'noise = {noise_level}'
    )
    
    with open('train_sim.py', 'w', encoding='utf-8') as f:
        f.write(patched)
        
    os.system('python train_sim.py')
    
    if os.path.exists('training_results.json'):
        os.replace('training_results.json', f'{model_name}_results.json')
    if os.path.exists('reward_curves.png'):
        os.replace('reward_curves.png', f'{model_name}_curves.png')

try:
    # 8B Simulation: Struggles (Noise 0.35)
    run_simulation("8B", "0.35")
    
    # 70B Simulation: Smart (Noise 0.05)
    run_simulation("70B", "0.05")
    
finally:
    if os.path.exists('train_sim.py'):
        os.remove('train_sim.py')
    print("Done! Check your folder for the new 8B and 70B files.")
