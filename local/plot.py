"""Plot training curves from EvalCallback logs."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot(log_dir: str = "logs"):
    eval_path = Path(log_dir) / "evaluations.npz"
    if not eval_path.exists():
        print(f"No eval data at {eval_path}. Run training first.")
        return

    data = np.load(eval_path)
    timesteps = data["timesteps"]
    results = data["results"]

    mean_r = results.mean(axis=1)
    std_r = results.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, mean_r, color="#2563eb", linewidth=2)
    ax.fill_between(timesteps, mean_r - std_r, mean_r + std_r, alpha=0.2, color="#2563eb")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("Humanoid-v5 Local Training")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    plot(args.log_dir)
