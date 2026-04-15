"""Evaluate and record a trained Humanoid-v5 agent."""

import argparse
from pathlib import Path

import yaml
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from stable_baselines3 import PPO


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(model_path: str, cfg: dict, n_episodes: int = 10, record: bool = True):
    video_dir = cfg["paths"]["videos"]
    Path(video_dir).mkdir(exist_ok=True)

    env = gym.make(
        cfg["env"]["id"],
        render_mode="rgb_array" if record else "human",
        **cfg["env"].get("kwargs", {}),
    )

    if record:
        env = RecordVideo(env, video_folder=video_dir, name_prefix="humanoid_eval",
                          episode_trigger=lambda ep: True)

    env = RecordEpisodeStatistics(env, buffer_length=n_episodes)
    model = PPO.load(model_path)

    rewards, lengths = [], []
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward, steps, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        rewards.append(total_reward)
        lengths.append(steps)
        print(f"Episode {ep + 1}/{n_episodes}: reward={total_reward:.1f}, steps={steps}")

    env.close()

    print(f"\n{'='*40}")
    print(f"Mean reward:  {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"Mean length:  {np.mean(lengths):.0f}")
    if record:
        print(f"Videos saved: {video_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Humanoid-v5")
    parser.add_argument("model", help="Path to trained model .zip")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    evaluate(args.model, load_config(args.config), args.episodes, not args.no_record)
