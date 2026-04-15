"""Measure distance traveled and height over a fixed time period.

Usage:
    python measure.py checkpoints/humanoid_ppo_final.zip
    python measure.py checkpoints/humanoid_ppo_50000000_steps.zip --duration 120
    python measure.py checkpoints/humanoid_ppo_50000000_steps.zip --episodes 5
"""

import argparse
from pathlib import Path

import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def measure(model_path: str, cfg: dict, vec_norm_path: str,
            duration_sec: float = 120.0, n_episodes: int = 3):
    env = make_vec_env(
        cfg["env"]["id"],
        n_envs=1,
        env_kwargs=cfg["env"].get("kwargs", {}),
    )
    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    inner_env = env.envs[0].unwrapped
    dt_per_step = inner_env.model.opt.timestep * inner_env.frame_skip
    max_steps = int(duration_sec / dt_per_step)

    print(f"Model: {model_path}")
    print(f"Duration: {duration_sec}s ({max_steps} steps at dt={dt_per_step:.4f}s)")
    print(f"Episodes: {n_episodes}")
    print()

    all_distances = []
    all_heights = []
    all_speeds = []
    all_survived = []
    all_rewards = []

    for ep in range(n_episodes):
        obs = env.reset()
        heights = []
        velocities = []
        total_reward = 0.0
        steps = 0
        last_info = {}

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            info = infos[0]
            total_reward += reward[0]
            steps += 1

            heights.append(inner_env.data.qpos[2])
            if "x_velocity" in info:
                velocities.append(info["x_velocity"])
            last_info = info

            if done[0]:
                break

        distance = last_info.get("x_position", 0.0)
        elapsed = steps * dt_per_step
        speed = distance / elapsed if elapsed > 0 else 0
        survived = steps == max_steps

        all_distances.append(distance)
        all_heights.append(heights)
        all_speeds.append(speed)
        all_survived.append(survived)
        all_rewards.append(total_reward)

        status = "SURVIVED" if survived else f"FELL at {elapsed:.1f}s"
        print(f"  Episode {ep+1}: distance={distance:.1f}m  speed={speed:.2f}m/s  "
              f"avg_velocity={np.mean(velocities):.2f}m/s  "
              f"height={np.mean(heights):.2f}m  reward={total_reward:.0f}  [{status}]")

    env.close()

    print(f"\n{'='*60}")
    print(f"SUMMARY ({n_episodes} episodes, {duration_sec}s each)")
    print(f"{'='*60}")
    print(f"  Distance:     {np.mean(all_distances):>8.1f}m  (+/- {np.std(all_distances):.1f})")
    print(f"  Speed:        {np.mean(all_speeds):>8.2f}m/s  (+/- {np.std(all_speeds):.2f})")
    print(f"  Height:       {np.mean([np.mean(h) for h in all_heights]):>8.2f}m  (avg torso z)")
    print(f"  Min height:   {np.mean([np.min(h) for h in all_heights]):>8.2f}m")
    print(f"  Max height:   {np.mean([np.max(h) for h in all_heights]):>8.2f}m")
    print(f"  Reward:       {np.mean(all_rewards):>8.0f}  (+/- {np.std(all_rewards):.0f})")
    print(f"  Survived:     {sum(all_survived)}/{n_episodes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure humanoid performance")
    parser.add_argument("model", help="Path to model .zip")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--duration", type=float, default=120.0, help="Duration in seconds (default 120)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--vec-normalize", default="checkpoints/vec_normalize.pkl")
    args = parser.parse_args()

    measure(args.model, load_config(args.config), args.vec_normalize,
            args.duration, args.episodes)
