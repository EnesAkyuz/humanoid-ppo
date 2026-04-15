"""Render videos from MJX/Brax checkpoints.

Runs the MJX environment on CPU JAX for correct observations,
then renders frames using MuJoCo's renderer.

Usage:
    python render_mjx.py checkpoints_v2_clean/step_766771200
    python render_mjx.py checkpoints_v2_clean/ --every 2
    python render_mjx.py checkpoints_v2_clean/ --episodes 5
"""

import argparse
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import yaml
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

# Force JAX to use CPU
jax.config.update("jax_platform_name", "cpu")

# Import the environment from train.py
sys.path.insert(0, str(Path(__file__).parent))
from train import HumanoidMJX


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_episode(env, inference_fn, mj_model, rng, max_steps=1000):
    """Run one episode using MJX env, render with MuJoCo."""
    renderer = mujoco.Renderer(mj_model, width=640, height=480)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance = 4.0
    cam.azimuth = 90
    cam.elevation = -20

    mj_data = mujoco.MjData(mj_model)
    state = env.reset(rng)
    frames = []
    total_reward = 0.0
    steps = 0

    for step in range(max_steps):
        # Get action from policy
        action, _ = inference_fn(state.obs, jax.random.PRNGKey(step))

        # Step MJX env
        state = env.step(state, action)
        total_reward += float(state.reward)
        steps += 1

        # Copy MJX state to CPU MuJoCo for rendering
        mj_data.qpos[:] = np.array(state.pipeline_state.qpos)
        mj_data.qvel[:] = np.array(state.pipeline_state.qvel)
        mujoco.mj_forward(mj_model, mj_data)

        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render())

        if float(state.done):
            break

    renderer.close()
    return frames, total_reward, steps


def save_video(frames, path, fps=30):
    import imageio
    imageio.mimsave(str(path), frames, fps=fps)
    print(f"  Saved: {path}")


def get_checkpoints(ckpt_dir, every=1):
    ckpts = []
    for p in Path(ckpt_dir).iterdir():
        if p.is_file() and p.name.startswith("step_"):
            match = re.search(r"step_(\d+)", p.name)
            if match:
                ckpts.append((int(match.group(1)), p))
    ckpts.sort(key=lambda x: x[0])
    if every > 1 and len(ckpts) > every:
        sampled = ckpts[::every]
        if ckpts[-1] not in sampled:
            sampled.append(ckpts[-1])
        ckpts = sampled
    return ckpts


def main():
    parser = argparse.ArgumentParser(description="Render MJX/Brax checkpoints")
    parser.add_argument("path", help="Checkpoint file or directory")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--every", type=int, default=1, help="Every Nth checkpoint")
    parser.add_argument("--output", default="videos", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video_dir = Path(args.output)
    video_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = cfg["env"]

    # Create MJX environment (runs on CPU via JAX)
    env = HumanoidMJX(
        forward_reward_weight=env_cfg["forward_reward_weight"],
        ctrl_cost_weight=env_cfg["ctrl_cost_weight"],
        healthy_reward=env_cfg["healthy_reward"],
        healthy_z_range=tuple(env_cfg["healthy_z_range"]),
        reset_noise_scale=env_cfg["reset_noise_scale"],
        exclude_current_positions_from_observation=env_cfg["exclude_current_positions_from_observation"],
    )

    # CPU MuJoCo model for rendering
    xml_path = str(Path(mujoco.__file__).parent / "mjx" / "test_data" / "humanoid" / "humanoid.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # Figure out obs/action sizes from env
    dummy_state = env.reset(jax.random.PRNGKey(0))
    obs_size = dummy_state.obs.shape[-1]
    action_size = env.action_size

    hidden_sizes = tuple(cfg["train"]["policy_hidden_layer_sizes"])

    # Get checkpoints
    path = Path(args.path)
    if path.is_dir():
        ckpts = get_checkpoints(path, every=args.every)
    else:
        match = re.search(r"step_(\d+)", path.name)
        step = int(match.group(1)) if match else 0
        ckpts = [(step, path)]

    if not ckpts:
        print(f"No checkpoints found at {args.path}")
        return

    # Build network once
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=action_size,
        policy_hidden_layer_sizes=hidden_sizes,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inference = ppo_networks.make_inference_fn(ppo_network)

    print(f"Rendering {len(ckpts)} checkpoint(s), {args.episodes} episodes each")
    print(f"Videos -> {video_dir}/\n")
    print(f"{'Checkpoint':<30} {'Reward':>10} {'Ep Length':>10}")
    print("-" * 52)

    # JIT compile the env step and reset
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    for step_num, ckpt_path in ckpts:
        params = model.load_params(str(ckpt_path))
        normalizer_state, policy_params, _ = params
        inference_fn = jax.jit(make_inference((normalizer_state, policy_params)))

        all_rewards = []
        all_lengths = []
        best_frames = None
        best_reward = -float("inf")

        for ep in range(args.episodes):
            rng = jax.random.PRNGKey(ep + 42)

            # Run episode manually with jitted functions
            state = jit_reset(rng)
            frames = []
            total_reward = 0.0
            ep_steps = 0
            mj_data = mujoco.MjData(mj_model)
            renderer = mujoco.Renderer(mj_model, width=640, height=480)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = 1
            cam.distance = 4.0
            cam.azimuth = 90
            cam.elevation = -20

            for s in range(1000):
                action, _ = inference_fn(state.obs, jax.random.PRNGKey(s))
                state = jit_step(state, action)
                total_reward += float(state.reward)
                ep_steps += 1

                mj_data.qpos[:] = np.array(state.pipeline_state.qpos)
                mj_data.qvel[:] = np.array(state.pipeline_state.qvel)
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera=cam)
                frames.append(renderer.render())

                if float(state.done):
                    break

            renderer.close()
            all_rewards.append(total_reward)
            all_lengths.append(ep_steps)

            if total_reward > best_reward:
                best_reward = total_reward
                best_frames = frames

        label = f"step_{step_num:09d}"
        video_path = video_dir / f"{label}.mp4"
        save_video(best_frames, video_path)

        mean_r = np.mean(all_rewards)
        mean_l = np.mean(all_lengths)
        print(f"{label:<30} {mean_r:>10.1f} {mean_l:>10.0f}")

    print(f"\nDone! Videos in {video_dir}/")


if __name__ == "__main__":
    main()
