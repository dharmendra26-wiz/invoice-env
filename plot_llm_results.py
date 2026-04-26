import json
import matplotlib.pyplot as plt

def plot_from_json(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Could not read {json_file}: {e}")
        return

    all_rewards = data.get("all_rewards", {})
    model_name = data.get("model", "unknown").split('/')[-1]
    
    if not all_rewards:
        print("No rewards found in json.")
        return

    tasks = ["easy", "medium", "hard", "expert_negotiation", "expert_fraud"]
    colors = ["#89b4fa", "#a6e3a1", "#f9e2af", "#fab387", "#f38ba8"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor("#11111b")
    fig.suptitle(f"Real LLM Benchmark: {model_name}", 
                 fontsize=16, fontweight="bold", color="#cdd6f4")
    axes = axes.flatten()

    for idx, (task, color) in enumerate(zip(tasks, colors)):
        ax = axes[idx]
        rewards = all_rewards.get(task, [])

        ax.set_facecolor("#1e1e2e")
        ax.set_title(task.replace("_", " ").title(),
                     fontsize=11, fontweight="bold", color="#cdd6f4", pad=8)
                     
        if not rewards:
            ax.text(0.5, 0.5, "No Data", color="#6c7086", ha="center", va="center", transform=ax.transAxes)
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
        ax.axhline(y=0.7, color="#a6e3a1", linestyle="--",
                   alpha=0.7, linewidth=1.3, label="Target 0.70")

        final_avg = sum(rewards[-max(5, win):]) / max(5, win)
        ax.annotate(f"Avg: {final_avg:.2f}",
                    xy=(len(rewards) - 1, final_avg),
                    xytext=(-40, 15), textcoords="offset points",
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0))

        ax.set_ylim(0, 1.1)
        ax.tick_params(colors="#6c7086", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#313244")
        ax.grid(True, alpha=0.15, color="#6c7086")
        ax.legend(fontsize=8, facecolor="#181825", labelcolor="#cdd6f4",
                  edgecolor="#313244", loc="lower right")

    axes[5].set_visible(False)
    
    out_file = f"{model_name}_real_curves.png".replace(":", "_").replace("-", "_")
    plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="#11111b")
    print(f"Reward curves saved -> {out_file}")

if __name__ == "__main__":
    import sys, glob
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        # Auto-detect most recently modified results JSON
        candidates = glob.glob("*_results.json")
        file = max(candidates, key=__import__("os").path.getmtime) if candidates else "training_results.json"
        print(f"Auto-detected results file: {file}")
    plot_from_json(file)
