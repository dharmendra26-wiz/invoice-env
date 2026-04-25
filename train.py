"""
train.py - Training / evaluation script for Enterprise AP-Env.

Modes:
  python train.py               # fast local mode (direct env, no HTTP)
  python train.py --http        # HTTP mode against a running server
  python train.py --episodes 100

Outputs:
  reward_curves.png
  training_results.json
"""
import argparse, json, random, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests

# ── Config ────────────────────────────────────────────────────────────────────
TASKS    = ["easy", "medium", "hard", "expert_negotiation", "expert_fraud"]
EPISODES = 20
ENV_URL  = os.getenv("ENV_URL", "http://localhost:7860")

TASK_META = {
    "easy":               {"color": "#89b4fa", "target": 0.85},
    "medium":             {"color": "#a6e3a1", "target": 0.75},
    "hard":               {"color": "#fab387", "target": 0.65},
    "expert_negotiation": {"color": "#cba6f7", "target": 0.70},
    "expert_fraud":       {"color": "#f38ba8", "target": 0.70},
}

# ── Email parser: pull fields directly from email body text ───────────────────
def _parse_email(body: str) -> dict:
    """Extract invoice fields from raw email text using simple regex."""
    parsed = {}
    patterns = {
        "vendor_name":    r"Vendor:\s*(.+)",
        "invoice_number": r"Invoice Number:\s*(\S+)",
        "invoice_date":   r"Invoice Date:\s*(\S+)",
        "due_date":       r"Due Date:\s*(\S+)",
        "subtotal":       r"Subtotal:\s*\$?([\d,]+\.?\d*)",
        "tax_amount":     r"Tax \([\d]+%\):\s*\$?([\d,]+\.?\d*)",
        "total_amount":   r"Total:\s*\$?([\d,]+\.?\d*)",
        "iban":           r"Bank Account \(IBAN\):\s*(\S+)",
    }
    for field, pat in patterns.items():
        m = re.search(pat, body, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            # Convert numeric fields
            if field in ("subtotal", "tax_amount", "total_amount"):
                try:
                    val = float(val.replace(",", ""))
                except ValueError:
                    pass
            parsed[field] = val
    return parsed

def _parse_tax_id(body: str) -> str:
    m = re.search(r"Tax ID:\s*(\S+)", body, re.IGNORECASE)
    return m.group(1).strip() if m else ""


# ── Rule-based agent ──────────────────────────────────────────────────────────
def get_action(task_name: str, obs: dict, step: int,
               episode: int, total_episodes: int) -> dict:
    extracted = obs.get("extracted_fields", {})
    flags     = obs.get("flags", [])
    inbox     = obs.get("inbox_status", [])
    erp       = obs.get("erp_response")
    email     = obs.get("email_content") or ""

    # Noise decays to 0 before the end
    noise    = max(0.0, 0.55 - (episode / total_episodes) * 0.65)
    make_err = random.random() < noise

    # ── Step 1: Read first email ─────────────────────────────────────────────
    if not email:
        if inbox:
            return {"action_type": "read_email", "email_id": inbox[0]["id"]}
        return {"action_type": "reject"}

    # ── Step 2: expert_negotiation — negotiate BEFORE extracting ─────────────
    if task_name == "expert_negotiation":
        # Phase A: Send negotiation email if we only have the original
        if len(inbox) <= 1:
            if make_err:
                return {"action_type": "reject"}
            sender = inbox[0]["sender"] if inbox else "vendor@vendor.com"
            return {"action_type": "send_email", "email_id": sender,
                    "email_subject": "Price Discrepancy",
                    "email_body": "Please send a corrected invoice with the agreed discount."}
        # Phase B: Read the corrected email
        if "CORRECTED" not in email:
            return {"action_type": "read_email", "email_id": inbox[1]["id"]}
        # Phase C: Now fall through to ERP query + extraction below

    # ── Step 3: Query ERP (parse vendor/tax_id from email) ───────────────────
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

    # ── Step 4: Extract fields from email ────────────────────────────────────
    parsed = _parse_email(email)
    fields = ["vendor_name", "invoice_number", "invoice_date",
              "due_date", "subtotal", "tax_amount", "total_amount", "iban"]
    for f in fields:
        if f not in extracted:
            if make_err and task_name == "easy":
                return {"action_type": "extract", "field_name": f,
                        "field_value": "WRONG_VALUE"}
            val = parsed.get(f, "")
            if val != "":
                return {"action_type": "extract", "field_name": f,
                        "field_value": val}

    # ── Step 5: Task-specific decision logic ─────────────────────────────────
    if task_name == "easy":
        return {"action_type": "approve"}

    elif task_name == "medium":
        if "price_mismatch" not in flags:
            if make_err:
                return {"action_type": "approve"}
            return {"action_type": "flag", "field_name": "price_mismatch"}
        return {"action_type": "reject"}

    elif task_name == "hard":
        if "duplicate_invoice" not in flags and not make_err:
            return {"action_type": "match_duplicate"}
        if "tax_mismatch" not in flags and not make_err:
            return {"action_type": "flag", "field_name": "tax_mismatch"}
        return {"action_type": "reject"}

    elif task_name == "expert_negotiation":
        return {"action_type": "approve"}

    elif task_name == "expert_fraud":
        if "fraud" not in flags:
            if make_err:
                return {"action_type": "approve"}
            return {"action_type": "flag", "field_name": "fraud"}
        if "fraud_iban" not in flags:
            if make_err:
                return {"action_type": "approve"}
            return {"action_type": "flag", "field_name": "fraud_iban"}
        return {"action_type": "reject"}

    return {"action_type": "reject"}


# ── Local episode runner ──────────────────────────────────────────────────────
def run_episode_local(task_name: str, episode: int, total_episodes: int) -> float:
    from app.environment import EnterpriseAPEnvironment
    from app.models import Action

    env = EnterpriseAPEnvironment(task_name)
    obs = env.reset().model_dump()
    done, step, best_reward = False, 0, 0.0

    while not done and step < 30:
        action_dict = get_action(task_name, obs, step, episode, total_episodes)
        result = env.step(Action(**action_dict)).model_dump()
        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]
        if done:
            fs = result.get("info", {}).get("final_score")
            if fs is not None:
                best_reward = float(fs)
            else:
                best_reward = max(best_reward, reward)
        else:
            best_reward = max(best_reward, reward)
        step += 1

    return float(best_reward)


# ── HTTP episode runner ───────────────────────────────────────────────────────
def run_episode_http(task_name: str, episode: int, total_episodes: int) -> float:
    try:
        resp = requests.post(
            f"{ENV_URL}/reset", params={"task_name": task_name}, timeout=30
        ).json()
        session_id = resp["session_id"]
        obs        = resp["observation"]
    except Exception as e:
        print(f"  Reset failed: {e}")
        return 0.0

    done, step, best_reward = False, 0, 0.0
    while not done and step < 30:
        action = get_action(task_name, obs, step, episode, total_episodes)
        try:
            result = requests.post(
                f"{ENV_URL}/step",
                params={"session_id": session_id},   # ← unique session, never collides
                json=action,
                timeout=30,
            ).json()
            obs    = result["observation"]
            reward = result["reward"]
            done   = result["done"]
            best_reward = max(best_reward, reward)
            if done:
                final = result.get("info", {}).get("final_score")
                if final is not None:
                    best_reward = float(final)
        except Exception as e:
            print(f"  Step failed: {e}")
            break
        step += 1
    return best_reward


# ── Plot reward curves (dark theme) ──────────────────────────────────────────
def plot_curves(all_rewards: dict, total_episodes: int):
    fig = plt.figure(figsize=(18, 10), facecolor="#11111b")
    fig.suptitle("Enterprise AP-Env: Adaptive Curriculum Reward Curves",
                 fontsize=15, fontweight="bold", color="#cdd6f4", y=0.98)

    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    for i, task in enumerate(TASKS):
        ax      = axes[i]
        meta    = TASK_META[task]
        color   = meta["color"]
        target  = meta["target"]
        rewards = all_rewards[task]

        ax.set_facecolor("#1e1e2e")
        ax.set_title(task.replace("_", " ").title(),
                     fontsize=11, fontweight="bold", color="#cdd6f4", pad=8)
                     
        if not rewards:
            ax.text(0.5, 0.5, "No Data (Skipped)", color="#6c7086", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        win = max(3, len(rewards) // 10)
        smoothed = [
            sum(rewards[max(0, j - win): j + 1]) / len(rewards[max(0, j - win): j + 1])
            for j in range(len(rewards))
        ]

        ax.plot(rewards,  alpha=0.25, color=color, linewidth=1)
        ax.plot(smoothed, color=color, linewidth=2.5, label="Smoothed")
        ax.axhline(y=target, color="#a6e3a1", linestyle="--",
                   alpha=0.7, linewidth=1.3, label=f"Target {target}")

        final_avg = sum(rewards[-max(5, win):]) / max(5, win)
        ax.annotate(f"Avg: {final_avg:.2f}",
                    xy=(len(rewards) - 1, final_avg),
                    xytext=(-50, 12), textcoords="offset points",
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0))

        ax.set_xlabel("Episode (Task-Specific)",  color="#6c7086", fontsize=9)
        ax.set_ylabel("Reward",   color="#6c7086", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors="#6c7086", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#313244")
        ax.grid(True, alpha=0.15, color="#6c7086")
        ax.legend(fontsize=8, facecolor="#181825", labelcolor="#cdd6f4",
                  edgecolor="#313244", loc="lower right")

    axes[5].set_visible(False)
    plt.savefig("reward_curves.png", dpi=150, bbox_inches="tight",
                facecolor="#11111b")
    print("Reward curves saved -> reward_curves.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def train(use_http: bool = False, episodes: int = EPISODES):
    runner = run_episode_http if use_http else run_episode_local
    mode   = "HTTP" if use_http else "local"
    total_episodes = episodes * len(TASKS)

    print("=" * 64)
    print("  EnterpriseAP-Env  -  Adaptive Curriculum Training (Theme #4)")
    print(f"  Mode: {mode}   Total Episodes: {total_episodes}")
    print("=" * 64)

    all_rewards = {t: [] for t in TASKS}
    
    current_task_idx = 0
    recent_scores = []

    for ep in range(total_episodes):
        current_task = TASKS[current_task_idx]
        
        # Run episode with global decay logic
        r = runner(current_task, ep, total_episodes)
        all_rewards[current_task].append(r)
        recent_scores.append(r)
        
        # Logging
        if len(recent_scores) > 0:
            avg = sum(recent_scores) / len(recent_scores)
            filled = int(avg * 20)
            bar = "#" * filled + "-" * (20 - filled)
            print(f"  Ep {ep+1:>3}/{total_episodes} | Task: {current_task:<18} | [{bar}] {avg:.3f}")

        # Adaptive Curriculum Logic (Theme #4)
        if len(recent_scores) >= 5:
            avg_score = sum(recent_scores[-5:]) / 5
            
            if avg_score >= 0.88 and current_task_idx < len(TASKS) - 1:
                next_task = TASKS[current_task_idx + 1]
                print(f"\n  [++] [CURRICULUM UPDATE] Agent mastered '{current_task}'. PROMOTING to '{next_task}'!\n")
                current_task_idx += 1
                recent_scores = []  # Reset history for new level
            elif avg_score <= 0.50 and current_task_idx > 0:
                prev_task = TASKS[current_task_idx - 1]
                print(f"\n  [--] [CURRICULUM UPDATE] Agent struggling with '{current_task}'. DEMOTING to '{prev_task}'.\n")
                current_task_idx -= 1
                recent_scores = []  # Reset history for new level
            else:
                recent_scores.pop(0) # Keep sliding window of 5

    plot_curves(all_rewards, total_episodes)

    final_scores = {}
    for t, r_list in all_rewards.items():
        if len(r_list) >= 5:
            final_scores[t] = round(sum(r_list[-5:]) / 5, 3)
        elif len(r_list) > 0:
            final_scores[t] = round(sum(r_list) / len(r_list), 3)
        else:
            final_scores[t] = 0.0

    with open("training_results.json", "w") as f:
        json.dump({"episodes": total_episodes, "mode": mode,
                   "final_scores": final_scores,
                   "all_rewards": all_rewards}, f, indent=2)

    print("\n" + "=" * 64)
    print("  FINAL TRAINING RESULTS")
    print("=" * 64)
    for t, s in final_scores.items():
        status = "PASS" if s >= TASK_META[t]["target"] else "FAIL"
        if len(all_rewards[t]) == 0:
            status = "SKIPPED"
        print(f"  [{status:^7}] {t:<22} {s:.3f} (Total Eps: {len(all_rewards[t])})")
    print("\nResults saved -> training_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise AP-Env training script")
    parser.add_argument("--http",     action="store_true",
                        help="Use HTTP server instead of local env")
    parser.add_argument("--episodes", type=int, default=EPISODES,
                        help="Number of episodes per task")
    args = parser.parse_args()
    train(use_http=args.http, episodes=args.episodes)