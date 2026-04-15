"""Humanoid-v5 PPO training — local CPU version (Stable-Baselines3)."""

import argparse
from pathlib import Path

import yaml
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env


ACTIVATION_MAP = {"ReLU": nn.ReLU, "Tanh": nn.Tanh}


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_policy_kwargs(cfg: dict) -> dict:
    pk = cfg.get("policy_kwargs", {})
    result = {}

    if "net_arch" in pk:
        result["net_arch"] = pk["net_arch"]
    if "activation_fn" in pk:
        result["activation_fn"] = ACTIVATION_MAP[pk["activation_fn"]]
    if "log_std_init" in pk:
        result["log_std_init"] = pk["log_std_init"]
    if "ortho_init" in pk:
        result["ortho_init"] = pk["ortho_init"]

    return result


def train(cfg: dict, resume: str | None = None):
    paths = cfg["paths"]
    for d in paths.values():
        Path(d).mkdir(exist_ok=True)

    tcfg = cfg["train"]
    use_normalize = tcfg.get("normalize", False)

    # Training env
    env = make_vec_env(
        cfg["env"]["id"],
        n_envs=cfg["env"]["n_envs"],
        seed=0,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=cfg["env"].get("kwargs", {}),
    )
    if use_normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=tcfg["gamma"])

    # Eval env
    eval_env = make_vec_env(
        cfg["env"]["id"],
        n_envs=1,
        seed=42,
        env_kwargs=cfg["env"].get("kwargs", {}),
    )
    if use_normalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, gamma=tcfg["gamma"], training=False)

    if resume:
        print(f"Resuming from {resume}")
        model = PPO.load(resume, env=env)
        model.learning_rate = lambda _: tcfg["learning_rate"]
        model.lr_schedule = lambda _: tcfg["learning_rate"]
        model.ent_coef = tcfg["ent_coef"]
        model.n_steps = tcfg["n_steps"]
        model.batch_size = tcfg["batch_size"]
        model.n_epochs = tcfg["n_epochs"]
        model.clip_range = lambda _: tcfg["clip_range"]
        model.max_grad_norm = tcfg["max_grad_norm"]
        print(f"  Updated: lr={tcfg['learning_rate']}, ent_coef={tcfg['ent_coef']}")
    else:
        policy_kwargs = build_policy_kwargs(tcfg)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=tcfg["learning_rate"],
            n_steps=tcfg["n_steps"],
            batch_size=tcfg["batch_size"],
            n_epochs=tcfg["n_epochs"],
            gamma=tcfg["gamma"],
            gae_lambda=tcfg["gae_lambda"],
            clip_range=tcfg["clip_range"],
            ent_coef=tcfg["ent_coef"],
            vf_coef=tcfg["vf_coef"],
            max_grad_norm=tcfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=paths["logs"],
            verbose=1,
            device="cpu",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(tcfg["total_timesteps"] // 20 // cfg["env"]["n_envs"], 1),
        save_path=paths["checkpoints"],
        name_prefix="humanoid_ppo",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=paths["checkpoints"],
        log_path=paths["logs"],
        eval_freq=cfg["eval"]["eval_freq"],
        n_eval_episodes=cfg["eval"]["n_eval_episodes"],
        deterministic=True,
    )

    model.learn(
        total_timesteps=tcfg["total_timesteps"],
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = Path(paths["checkpoints"]) / "humanoid_ppo_final"
    model.save(final_path)
    if use_normalize:
        env.save(Path(paths["checkpoints"]) / "vec_normalize.pkl")
    print(f"Done. Model saved to {final_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Humanoid-v5 (local CPU)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .zip")
    args = parser.parse_args()

    train(load_config(args.config), resume=args.resume)
