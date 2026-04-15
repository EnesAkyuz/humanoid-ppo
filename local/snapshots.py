"""Record evaluation videos at regular checkpoint intervals.

Uses VecNormalize stats for correct policy behavior, renders with imageio.

Usage:
    python snapshots.py                    # auto-detect all checkpoints
    python snapshots.py --every 3          # every 3rd checkpoint
    python snapshots.py --best-only        # just the best model
"""

import argparse
import re
from pathlib import Path

import yaml
import numpy as np
import gymnasium as gym
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_checkpoints(ckpt_dir: str, every: int = 1) -> list[tuple[int, Path]]:
    ckpts = []
    for p in Path(ckpt_dir).glob("humanoid_ppo_*_steps.zip"):
        match = re.search(r"_(\d+)_steps", p.name)
        if match:
            ckpts.append((int(match.group(1)), p))

    ckpts.sort(key=lambda x: x[0])

    if every > 1 and len(ckpts) > every:
        sampled = ckpts[::every]
        if ckpts[-1] not in sampled:
            sampled.append(ckpts[-1])
        ckpts = sampled

    best = Path(ckpt_dir) / "best_model.zip"
    if best.exists():
        ckpts.append((-1, best))

    final = Path(ckpt_dir) / "humanoid_ppo_final.zip"
    if final.exists():
        ckpts.append((-2, final))

    return ckpts


def record_snapshot(model_path: Path, video_dir: Path, label: str,
                    cfg: dict, vec_norm_path: str, n_episodes: int = 2):
    """Record episodes with proper VecNormalize and rendering."""
    # Create VecEnv with VecNormalize for correct observations
    vec_env = make_vec_env(
        cfg["env"]["id"],
        n_envs=1,
        env_kwargs=cfg["env"].get("kwargs", {}),
    )
    vec_env = VecNormalize.load(vec_norm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)

    # Separate render env (same physics, with rgb rendering)
    render_env = gym.make(
        cfg["env"]["id"],
        render_mode="rgb_array",
        **cfg["env"].get("kwargs", {}),
    )

    rewards = []
    lengths = []
    best_frames = None
    best_reward = -float("inf")

    for ep in range(n_episodes):
        # Reset both envs with same seed
        obs = vec_env.reset()
        render_obs, _ = render_env.reset(seed=ep + 100)

        # Sync the render env state to match vec_env
        inner_env = vec_env.envs[0].unwrapped
        render_inner = render_env.unwrapped
        render_inner.data.qpos[:] = inner_env.data.qpos[:]
        render_inner.data.qvel[:] = inner_env.data.qvel[:]
        render_inner.model.opt.timestep = inner_env.model.opt.timestep

        frames = []
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            # Step the normalized env (for correct obs)
            obs, reward, dones, infos = vec_env.step(action)
            total_reward += reward[0]
            steps += 1
            done = dones[0]

            # Step the render env with same action and sync state
            render_env.step(action[0])
            render_inner.data.qpos[:] = inner_env.data.qpos[:]
            render_inner.data.qvel[:] = inner_env.data.qvel[:]
            import mujoco
            mujoco.mj_forward(render_inner.model, render_inner.data)

            frame = render_env.render()
            frames.append(frame)

        rewards.append(total_reward)
        lengths.append(steps)

        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames

    vec_env.close()
    render_env.close()

    # Save best episode as video
    if best_frames:
        video_path = video_dir / f"{label}.mp4"
        imageio.mimsave(str(video_path), best_frames, fps=30)
        print(f"  Saved: {video_path}")

    return np.mean(rewards), np.mean(lengths)


def main():
    parser = argparse.ArgumentParser(description="Record training progression videos")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--every", type=int, default=1, help="Record every Nth checkpoint")
    parser.add_argument("--best-only", action="store_true", help="Only record best model")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per checkpoint")
    parser.add_argument("--vec-normalize", default=None, help="Path to VecNormalize .pkl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video_dir = Path("videos/snapshots")
    video_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect vec_normalize.pkl
    vec_norm_path = args.vec_normalize
    if vec_norm_path is None:
        default_path = Path(args.ckpt_dir) / "vec_normalize.pkl"
        if default_path.exists():
            vec_norm_path = str(default_path)

    if not vec_norm_path or not Path(vec_norm_path).exists():
        print("ERROR: No vec_normalize.pkl found. Training must finish first.")
        print("       Or pass --vec-normalize path/to/stats.pkl")
        return

    print(f"Using VecNormalize stats: {vec_norm_path}")

    if args.best_only:
        best = Path(args.ckpt_dir) / "best_model.zip"
        if not best.exists():
            print(f"No best_model.zip in {args.ckpt_dir}/")
            return
        ckpts = [(-1, best)]
    else:
        ckpts = get_checkpoints(args.ckpt_dir, every=args.every)

    if not ckpts:
        print(f"No checkpoints found in {args.ckpt_dir}/")
        return

    print(f"Recording {len(ckpts)} checkpoints, {args.episodes} episodes each")
    print(f"Videos -> {video_dir}/\n")
    print(f"{'Checkpoint':<40} {'Reward':>10} {'Ep Length':>10}")
    print("-" * 62)

    for steps, path in ckpts:
        if steps == -1:
            label = "best_model"
        elif steps == -2:
            label = "final"
        else:
            label = f"step_{steps:08d}"

        mean_r, mean_len = record_snapshot(path, video_dir, label, cfg, vec_norm_path, args.episodes)
        print(f"{label:<40} {mean_r:>10.1f} {mean_len:>10.0f}")

    print(f"\nDone! Videos in {video_dir}/")


if __name__ == "__main__":
    main()
