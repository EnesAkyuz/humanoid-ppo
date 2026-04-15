"""Humanoid PPO training — MJX GPU-accelerated (Brax + MuJoCo XLA).

Runs thousands of parallel physics simulations on GPU via MJX,
with Brax's JIT-compiled PPO training loop. Everything stays on-device.
"""

import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import yaml

from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class HumanoidMJX(PipelineEnv):
    """Humanoid environment matching Gymnasium Humanoid-v5 rewards, running on MJX."""

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        # Load the standard MuJoCo Humanoid model
        mj_model = mujoco.MjModel.from_xml_path(
            str(Path(mujoco.__file__).parent / "mjx" / "test_data" / "humanoid" / "humanoid.xml")
        )
        # Solver settings optimized for MJX
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        kwargs["n_frames"] = kwargs.get("n_frames", 5)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_min = healthy_z_range[0]
        self._healthy_z_max = healthy_z_range[1]
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions = exclude_current_positions_from_observation

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=low, maxval=hi
        )

        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data, jp.zeros(self.sys.nu))

        reward, done = jp.zeros(2)
        metrics = {
            "forward_reward": jp.zeros(()),
            "ctrl_cost": jp.zeros(()),
            "healthy_reward": jp.zeros(()),
            "upright_reward": jp.zeros(()),
            "symmetry_cost": jp.zeros(()),
            "arm_penalty": jp.zeros(()),
            "abdomen_penalty": jp.zeros(()),
        }

        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        # Forward reward: x-velocity of center of mass
        com_before = data0.subtree_com[1]
        com_after = data.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]

        # Control cost (doubled to discourage flailing)
        ctrl_cost = self._ctrl_cost_weight * 2.0 * jp.sum(jp.square(action))

        # Healthy reward
        z = data.qpos[2]
        is_healthy = jp.where(z < self._healthy_z_min, 0.0, 1.0)
        is_healthy = jp.where(z > self._healthy_z_max, 0.0, is_healthy)
        healthy_reward = self._healthy_reward * is_healthy

        # --- Gait shaping rewards ---

        # Upright torso: quaternion qpos[3:7].
        # up_z = 1 when perfectly upright, -1 when upside down
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        torso_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
        # Also penalize forward lean: forward_x should be ~0 when upright
        torso_forward_x = 2.0 * (qx*qz + qw*qy)
        upright_reward = 5.0 * torso_up_z - 3.0 * jp.abs(torso_forward_x)

        # Arms should hang down, not stretch out.
        # Penalize shoulder joints being far from 0 (neutral hanging position)
        # shoulder1_right: qpos[22], shoulder2_right: qpos[23], elbow_right: qpos[24]
        # shoulder1_left: qpos[25], shoulder2_left: qpos[26], elbow_left: qpos[27]
        arm_joints = data.qpos[22:28]
        arm_penalty = 2.0 * jp.sum(jp.square(arm_joints))

        # Symmetry: penalize difference between left and right leg joints
        leg_right = data.qpos[10:16]
        leg_left = data.qpos[16:22]
        symmetry_cost = 1.0 * jp.sum(jp.square(leg_right - leg_left))

        # Abdomen should stay near neutral (no extreme bending)
        # abdomen_z: qpos[7], abdomen_y: qpos[8], abdomen_x: qpos[9]
        abdomen_joints = data.qpos[7:10]
        abdomen_penalty = 2.0 * jp.sum(jp.square(abdomen_joints))

        reward = (forward_reward + healthy_reward + upright_reward
                  - ctrl_cost - symmetry_cost - arm_penalty - abdomen_penalty)
        done = 1.0 - is_healthy

        obs = self._get_obs(data, action)

        state.metrics.update(
            forward_reward=forward_reward,
            ctrl_cost=ctrl_cost,
            healthy_reward=healthy_reward,
            upright_reward=upright_reward,
            symmetry_cost=symmetry_cost,
            arm_penalty=arm_penalty,
            abdomen_penalty=abdomen_penalty,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, data: mjx.Data, action: jax.Array) -> jax.Array:
        position = data.qpos
        if self._exclude_current_positions:
            position = position[2:]

        return jp.concatenate([
            position,
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            data.qfrc_actuator,
        ])


def train(cfg: dict, resume: str | None = None):
    paths = cfg["paths"]
    for d in paths.values():
        Path(d).mkdir(exist_ok=True)

    env_cfg = cfg["env"]
    tcfg = cfg["train"]

    # Register and create environment
    envs.register_environment("humanoid_mjx", HumanoidMJX)
    env = envs.get_environment(
        "humanoid_mjx",
        forward_reward_weight=env_cfg["forward_reward_weight"],
        ctrl_cost_weight=env_cfg["ctrl_cost_weight"],
        healthy_reward=env_cfg["healthy_reward"],
        healthy_z_range=tuple(env_cfg["healthy_z_range"]),
        reset_noise_scale=env_cfg["reset_noise_scale"],
        exclude_current_positions_from_observation=env_cfg["exclude_current_positions_from_observation"],
    )

    # Network factory
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(tcfg["policy_hidden_layer_sizes"]),
    )

    # Training function
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=tcfg["num_timesteps"],
        num_evals=tcfg["num_evals"],
        reward_scaling=tcfg["reward_scaling"],
        episode_length=env_cfg["episode_length"],
        normalize_observations=tcfg["normalize_observations"],
        action_repeat=1,
        unroll_length=tcfg["unroll_length"],
        num_minibatches=tcfg["num_minibatches"],
        num_updates_per_batch=tcfg["num_updates_per_batch"],
        discounting=tcfg["discounting"],
        learning_rate=tcfg["learning_rate"],
        entropy_cost=tcfg["entropy_cost"],
        clipping_epsilon=tcfg["clipping_epsilon"],
        max_grad_norm=tcfg.get("max_grad_norm"),
        gae_lambda=tcfg["gae_lambda"],
        num_envs=tcfg["num_envs"],
        batch_size=tcfg["batch_size"],
        seed=tcfg["seed"],
        network_factory=make_networks_factory,
    )

    # Progress callback with periodic checkpointing
    metrics_history = []
    start_time = time.time()
    ckpt_dir = Path(paths["checkpoints"])
    saved_params = {}

    def policy_params_fn(step, make_policy, params):
        """Called by Brax with current params at each eval — save a checkpoint."""
        saved_params["latest"] = params
        saved_params["latest_step"] = step
        ckpt_path = ckpt_dir / f"step_{int(step):09d}"
        model.save_params(str(ckpt_path), params)
        print(f"    [checkpoint saved: {ckpt_path}]")

    def progress_fn(step, metrics):
        elapsed = time.time() - start_time
        reward = metrics.get("eval/episode_reward", 0)
        sps = step / elapsed if elapsed > 0 else 0

        metrics_history.append({"step": int(step), "reward": float(reward), "elapsed": elapsed})
        print(f"  step={step:>12,}  reward={reward:>8.1f}  sps={sps:>10,.0f}  elapsed={elapsed:>6.1f}s")

    # Resume support: load existing params and pass as init
    resume_params = None
    if resume:
        print(f"Loading checkpoint: {resume}")
        resume_params = model.load_params(resume)
        print(f"  Loaded. Will fine-tune with new reward function.")

    print(f"Device: {jax.devices()[0]}")
    print(f"Parallel envs: {tcfg['num_envs']}")
    print(f"Total timesteps: {tcfg['num_timesteps']:,}")
    print(f"Training...")
    print()

    train_kwargs = dict(
        environment=env,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )
    if resume_params is not None:
        train_kwargs["restore_params"] = resume_params

    make_inference_fn, params, final_metrics = train_fn(**train_kwargs)

    # Save model
    ckpt_path = Path(paths["checkpoints"]) / "mjx_humanoid_final"
    model.save_params(str(ckpt_path), params)
    print(f"\nModel saved to {ckpt_path}")

    # Save metrics
    metrics_path = Path(paths["logs"]) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    if metrics_history:
        print(f"Final reward: {metrics_history[-1]['reward']:.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Humanoid (MJX + Brax PPO)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(load_config(args.config), resume=args.resume)
