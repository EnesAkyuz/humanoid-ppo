# Humanoid-v5 PPO Training

Training a MuJoCo Humanoid-v5 to walk using PPO, with both local CPU and Lambda GPU pipelines.

## Results Summary

| Run | Framework | Device | Peak Reward | Ep Length | Steps | Wall Time |
|-----|-----------|--------|-------------|-----------|-------|-----------|
| Local v1 (default config) | SB3 | M4 Mac CPU | ~650 (plateau) | ~125 | 40M | ~12 hrs |
| Local v2 (Zoo config) | SB3 + VecNormalize | M4 Mac CPU | 5,540+ | 536 | 10M | ~45 min |
| Lambda v1 (MJX, lr=3e-4) | Brax + MJX | A100 40GB | 6,434 | — | 79M | ~12 min |
| Lambda v2 (MJX, lr=1e-4) | Brax + MJX | A100 40GB | **18,052** | 737 | 766M | ~80 min |
| Lambda v3 (gait shaping v1) | Brax + MJX | A100 40GB | 25,982 | 1000 | 20M | ~6 min |
| Lambda v4 (gait shaping v2) | Brax + MJX | A100 40GB | 18,715 | 1000 | 51M | ~10 min |

## Technical Details

### Hardware

**Local**: Apple M4 MacBook Pro
- CPU-only training (MPS not used by SB3)
- ~3,500 steps/sec with 8 parallel envs
- Python 3.11, Stable-Baselines3 2.8.0

**Lambda Cloud**: gpu_1x_a100_sxm4
- NVIDIA A100 40GB SXM4, 30 vCPUs, 200 GiB RAM
- $1.99/hr, us-east-1 region
- ~130,000 steps/sec with 4096 parallel envs (MJX)
- Python 3.10, JAX 0.6.2, Brax 0.14.1, MuJoCo MJX 3.6.0
- Persistent filesystem: user-created, mounted at `$LAMBDA_FILESYSTEM_MOUNT` (see `.env.example`)

### Environment

- **Task**: MuJoCo Humanoid-v5 — 21 actuators, 376-dim observation space
- **Physics**: 5 substeps per control step (n_frames=5)
- **Episode**: max 1000 steps, terminates if torso height leaves [1.0, 2.0]
- **Observation**: qpos (excl. root xy), qvel, cinert, cvel, qfrc_actuator (336 dims in MJX)

### Architecture

**Local (SB3)**:
- Separate actor/critic MLP networks: pi=[256,256], vf=[256,256]
- ReLU activation (not default Tanh)
- log_std_init=-2 (low initial exploration)
- ortho_init=false
- VecNormalize on observations and rewards

**Lambda (MJX/Brax)**:
- Shared MLP policy: [256, 256] hidden layers
- Brax default activations
- Running statistics observation normalization
- reward_scaling=0.05

### Training Pipeline

**Two separate pipelines**:

1. **Local (SB3 + CPU MuJoCo)**: Standard Stable-Baselines3 PPO with SubprocVecEnv. Environments run on CPU cores. Good for iteration, slow for large-scale training.

2. **Lambda (Brax + MJX on GPU)**: MuJoCo XLA compiles physics to GPU via JAX. Entire training loop (env stepping, rollout collection, GAE, minibatch SGD) is JIT-compiled and runs on-device. 4096 parallel environments batched on GPU. ~37x faster than local.

### PPO Hyperparameters

**Local (Zoo-tuned)**:
```yaml
learning_rate: 3.57e-5
n_steps: 512
batch_size: 256
n_epochs: 5
gamma: 0.95
gae_lambda: 0.9
clip_range: 0.3
ent_coef: 0.00238
vf_coef: 0.43
max_grad_norm: 2
```

**Lambda (MJX)**:
```yaml
learning_rate: 3.0e-5      # v4, was 3e-4 in v1
num_envs: 4096
unroll_length: 10
batch_size: 512
num_minibatches: 24
num_updates_per_batch: 8
discounting: 0.97
gae_lambda: 0.95
entropy_cost: 1.0e-3
clipping_epsilon: 0.3
max_grad_norm: 1.0          # added in v4 to prevent NaN
reward_scaling: 0.05        # reduced in v4 from 0.1
```

### Reward Function Evolution

**v1-v2 (vanilla)**: Standard Humanoid-v5 reward
```
reward = forward_velocity * 1.25 + healthy_reward * 5.0 - ctrl_cost * 0.1
```

**v3 (gait shaping)**: Added upright torso bonus and symmetry penalty
```
+ upright_reward (2.0 * torso_up_z)
- symmetry_cost (left/right joint difference)
- ctrl_cost doubled
```

**v4 (stronger gait shaping)**: Addressed forward lean and arm stretching
```
+ upright_reward (5.0 * torso_up_z - 3.0 * |forward_lean|)
- arm_penalty (2.0 * shoulder/elbow joints squared)
- abdomen_penalty (2.0 * waist joints squared)
- symmetry_cost (leg joints only)
- ctrl_cost doubled
```

### Checkpoint Format

- **Local (SB3)**: `.zip` files loaded with `PPO.load()`. VecNormalize stats saved as `vec_normalize.pkl`.
- **Lambda (Brax)**: Binary files saved with `brax.io.model.save_params()`. Contains tuple of `(normalizer_state, policy_params, value_params)`. Loaded with `brax.io.model.load_params()`.

**Important**: MJX checkpoints cannot be rendered with raw CPU MuJoCo observations — the `cinert`/`cvel` values differ between MJX and CPU MuJoCo. The `render_mjx.py` script runs the MJX environment on CPU JAX for correct observations, then copies qpos/qvel to CPU MuJoCo for rendering.

## Key Findings

### VecNormalize is critical
The single biggest improvement came from adding `VecNormalize` (observation + reward normalization). Without it, PPO plateaus at ~650 reward on Humanoid. With it, the same architecture reaches 5,000+ easily.

### Zoo hyperparameters matter
The Optuna-tuned config from rl-baselines3-zoo made a massive difference vs hand-tuned defaults. Key differences: much lower learning rate (3.57e-5 vs 3e-4), lower gamma (0.95 vs 0.99), separate actor/critic nets, ReLU activation, low initial std.

### MJX speed vs SB3 efficiency
SB3 with VecNormalize is more sample-efficient (higher reward per step) but much slower in wall time. MJX is less efficient per step but compensates with raw throughput. For experimentation and iteration, MJX wins.

### NaN divergence is a recurring problem
Training diverged to NaN in 3 out of 4 MJX runs, typically after 40-80M steps of fine-tuning. Mitigations that helped: `max_grad_norm: 1.0`, lower `reward_scaling` (0.05), lower learning rate. The root cause is likely reward magnitudes growing large enough to cause gradient explosion.

### Reward hacking is real
The humanoid consistently found exploitative gaits (leaning forward, stretching arms behind, lunging with one leg) that maximize forward velocity reward without natural walking. Reward shaping (upright bonus, arm/abdomen penalties, symmetry) improved posture incrementally but didn't fully solve it. Natural walking likely requires reference motion imitation (DeepMimic-style).

## Project Structure

Tracked in git:

```
.
├── README.md                  # narrative overview + embedded videos
├── EXPLANATION.md             # beginner-friendly walkthrough of how PPO learns
├── docs/
│   └── TECHNICAL.md           # this file — the dense reference
├── config.yaml                # Lambda MJX config (latest: v4 with gait shaping)
├── train.py                   # Lambda MJX training (Brax + MJX + reward shaping)
├── render_mjx.py              # Render videos from MJX checkpoints
├── requirements.txt           # Lambda deps (jax, brax, mujoco-mjx)
├── check_availability.sh      # Check Lambda GPU availability
├── lambda_run.sh              # Auto-launch Lambda instance + train
├── plot.py                    # Plot training curves
├── evaluate.py                # Evaluate SB3 models
├── .env.example               # template — copy to .env and fill in locally
├── videos/                    # Local SB3 early progression (pre-Zoo config)
├── videos_v2/                 # Lambda v2 progression (vanilla reward)
├── videos_v3/                 # Lambda v3 progression (gait shaping v1)
├── videos_v4/                 # Lambda v4 progression (gait shaping v2)
├── training_v2_progression.png
├── training_v2_full_progression.png
└── local/
    ├── config.yaml            # Local SB3 config (Zoo hyperparams + VecNormalize)
    ├── train.py               # Local SB3 training
    ├── evaluate.py            # Evaluate locally
    ├── snapshots.py           # Record progression videos (SB3)
    ├── measure.py             # Ad-hoc reward measurement
    ├── plot.py                # Plot local training curves
    ├── requirements.txt       # Local deps (stable-baselines3)
    ├── CHANGES.md             # Hyperparameter change log
    ├── videos/snapshots/      # Local SB3 progression snapshots
    └── training_progress.png
```

Gitignored (each person generates these locally):

```
.env                           # your secrets (API key, SSH key name, filesystem name)
.venv/, local/.venv/           # virtual environments
checkpoints*/                  # trained model weights
local/checkpoints/             # local SB3 weights
local/logs/                    # TensorBoard event files
```

## Lambda Cloud Setup

- **Instance**: gpu_1x_a100_sxm4 ($1.99/hr), us-east-1
- **Filesystem**: create one at `cloud.lambda.ai` → Filesystems (must be in the same region as the instance). Record its name/ID in `.env` as `LAMBDA_FILESYSTEM_NAME` / `LAMBDA_FILESYSTEM_ID`.
- **Mount**: `/lambda/nfs/<your-filesystem-name>/humanoid`
- **SSH key**: add one at `cloud.lambda.ai` → SSH Keys, record its name in `.env` as `SSH_KEY_NAME`.
- **Venv on filesystem**: `<mount>/humanoid/.venv` (persists between instances)

### Launch and train
```bash
# Check availability
bash check_availability.sh

# SSH in (after launching instance with filesystem attached in us-east-1)
source .env && ssh -i $SSH_KEY_PATH ubuntu@<IP>

# On instance
cd "$LAMBDA_FILESYSTEM_MOUNT/humanoid"
source .venv/bin/activate
python3 train.py                                    # fresh start
python3 train.py --resume checkpoints/step_XXXXXX   # resume from checkpoint
```

### Pull checkpoints locally
```bash
source .env && rsync -az -e "ssh -i $SSH_KEY_PATH" \
    "ubuntu@<IP>:$LAMBDA_FILESYSTEM_MOUNT/humanoid/checkpoints/" checkpoints_vN/
```

## Rendering Videos

Requires the root `.venv` (JAX + Brax + MuJoCo):
```bash
cd ~/Humanoid && source .venv/bin/activate

# Best v4 model
python render_mjx.py checkpoints_v4/step_051609600

# Full progression of any version
python render_mjx.py checkpoints_v4/ --output videos_v4

# Every 3rd checkpoint, 5 episodes each
python render_mjx.py checkpoints_v2_clean/ --every 3 --episodes 5
```

## Local Training

Uses `local/.venv` (SB3 + PyTorch):
```bash
cd ~/Humanoid/local && source .venv/bin/activate
caffeinate -i python3 train.py                                    # fresh start
caffeinate -i python3 train.py --resume checkpoints/<ckpt>.zip    # resume
```

## Next Steps

- **Fix NaN divergence**: Try lower reward_scaling (0.01), stronger gradient clipping, or learning rate warmup
- **Natural walking**: Implement DeepMimic-style reference motion imitation instead of hand-crafted reward shaping
- **MuJoCo Playground**: Switch from deprecated Brax to Google's mujoco_playground for better-maintained MJX environments with pre-built locomotion tasks
- **Curriculum learning**: Start with standing, then slow walking, then fast walking — gradually increase forward_reward_weight
