# humanoid-ppo

Teaching a MuJoCo humanoid to walk with PPO, studied as **two
contrasting case studies**:

1. **[Local SB3 + CPU](#case-study-1--local-sb3--cpu-mac-m4)** — the
   headline result. Much fewer total steps, much slower per step, but
   ends with the **cleanest, most natural-looking walking gait** of any
   run in this project.
2. **[Lambda Brax + MJX + A100](#case-study-2--lambda-brax--mjx--a100)**
   — 37× faster iteration, hits higher raw reward numbers, and invents
   a hilarious reward-hacking "speed-skater" gait along the way.

Same task, same algorithm family (PPO), wildly different trajectories.
The moral: **fastest training is not best training.**

<p align="center">
  <img src="local/videos/snapshots/step_50000000.gif" width="560" loading="lazy" alt="step_50000000">
</p>
<p align="center"><em>Local SB3 policy at 50M steps — the cleanest gait we got, from ~4 hours of M4 CPU training.</em></p>

📖 Plain-English walkthrough of how PPO learns and what we tried:
**[EXPLANATION.md](EXPLANATION.md)**

📘 Technical reference (hyperparameters, architecture, pipeline):
**[docs/TECHNICAL.md](docs/TECHNICAL.md)**

---

## TL;DR

- **The local CPU run — 50M steps in ~4 hours — ends with the
  best-looking gait of any run here.** Not the highest raw reward
  number, just the one that actually looks like a human walking.
- MacBook CPU with default config plateau'd at ~650 reward for 35M
  steps. `VecNormalize` + Zoo-tuned hyperparameters broke the plateau
  in minutes and took the policy to **10,397 peak reward** / ~7,000
  smoothed with a clean gait.
- Switching to Brax + MJX on an A100 was a **37× speedup**. 50M steps
  went from ~4 hours to ~6 minutes, which made reward-function
  iteration practical.
- The MJX runs do hit higher raw numbers (**18,052 peak**) but via a
  "speed-skater" gait — leaning forward, arms back, one leg as a
  rudder. Classic reward hacking. v3/v4 reward shaping improved
  posture but the policy kept finding new loopholes.

### Results summary

| Run | Framework | Device | Peak Reward | Ep. Length | Steps | Wall Time |
|---|---|---|---|---|---|---|
| Local v1 (default config) | SB3 | M4 CPU | ~650 (plateau) | ~125 | 40M | ~12 h |
| **Local v2 (Zoo config)** ⭐ | **SB3 + VecNormalize** | **M4 CPU** | **10,397** | **578** | **50M** | **~4 h** |
| Lambda v1 (lr 3e-4) | Brax + MJX | A100 | 6,434 | — | 79M | ~12 min |
| Lambda v2 (lr 1e-4) | Brax + MJX | A100 | 18,052 | 737 | 766M | ~80 min |
| Lambda v3 (gait shaping v1) | Brax + MJX | A100 | 25,982 | 1000 | 20M | ~6 min |
| Lambda v4 (gait shaping v2) | Brax + MJX | A100 | 18,715 | 1000 | 51M | ~10 min |

⭐ = best-looking gait. _(Lambda rewards aren't directly comparable to
local — Lambda v3+ adds shaping bonuses that inflate the headline
number, and MJX's running-stats normalizer scales things differently
from `VecNormalize`.)_

### Training curves

Local first — the Zoo-config run that produced the headline gait:

![Local training progression](local/training_progress.png)

Then the Lambda v2 run for contrast — many more steps, higher raw
reward, worse-looking gait:

![Lambda v2 full progression](training_v2_full_progression.png)

---

## Case Study 1 — Local SB3 + CPU (Mac M4)

Stable-Baselines3 2.8.0 + PyTorch, 8 CPU cores on an M4 MacBook Pro.
~3,500 env steps/sec. Slower per step than the GPU pipeline, but
**much more sample-efficient** — this run reached a cleaner walking
gait than any of the Lambda MJX runs despite using less than 1/15th
the total steps.

### Training progression (post-Zoo-config fix)

Snapshots every ~2.5M steps from init to 50M.

<table>
  <tr>
    <td align="center"><sub>2.5M steps</sub><br>
      <img src="local/videos/snapshots/step_02500000.gif" width="260" loading="lazy" alt="step_02500000">
    </td>
    <td align="center"><sub>10M steps</sub><br>
      <img src="local/videos/snapshots/step_10000000.gif" width="260" loading="lazy" alt="step_10000000">
    </td>
    <td align="center"><sub>17.5M steps</sub><br>
      <img src="local/videos/snapshots/step_17500000.gif" width="260" loading="lazy" alt="step_17500000">
    </td>
  </tr>
  <tr>
    <td align="center"><sub>25M steps</sub><br>
      <img src="local/videos/snapshots/step_25000000.gif" width="260" loading="lazy" alt="step_25000000">
    </td>
    <td align="center"><sub>32.5M steps</sub><br>
      <img src="local/videos/snapshots/step_32500000.gif" width="260" loading="lazy" alt="step_32500000">
    </td>
    <td align="center"><sub>40M steps</sub><br>
      <img src="local/videos/snapshots/step_40000000.gif" width="260" loading="lazy" alt="step_40000000">
    </td>
  </tr>
  <tr>
    <td align="center"><sub>47.5M steps</sub><br>
      <img src="local/videos/snapshots/step_47500000.gif" width="260" loading="lazy" alt="step_47500000">
    </td>
    <td align="center"><sub><strong>50M steps — headline gait</strong></sub><br>
      <img src="local/videos/snapshots/step_50000000.gif" width="260" loading="lazy" alt="step_50000000">
    </td>
    <td></td>
  </tr>
</table>

### The pre-Zoo plateau run

The first local run — **default hyperparameters, no `VecNormalize`**.
Plateau'd at ~650 reward and never broke out. Watch the shuffle never
become a walk, even after 185M steps.

<table>
  <tr>
    <td align="center"><sub>0 steps (random init)</sub><br>
      <img src="videos/step_000000000.gif" width="260" loading="lazy" alt="step_000000000">
    </td>
    <td align="center"><sub>26M steps</sub><br>
      <img src="videos/step_026419200.gif" width="260" loading="lazy" alt="step_026419200">
    </td>
    <td align="center"><sub>53M steps</sub><br>
      <img src="videos/step_052838400.gif" width="260" loading="lazy" alt="step_052838400">
    </td>
  </tr>
  <tr>
    <td align="center"><sub>79M steps</sub><br>
      <img src="videos/step_079257600.gif" width="260" loading="lazy" alt="step_079257600">
    </td>
    <td align="center"><sub>106M steps</sub><br>
      <img src="videos/step_105676800.gif" width="260" loading="lazy" alt="step_105676800">
    </td>
    <td align="center"><sub>132M steps</sub><br>
      <img src="videos/step_132096000.gif" width="260" loading="lazy" alt="step_132096000">
    </td>
  </tr>
  <tr>
    <td align="center"><sub>159M steps</sub><br>
      <img src="videos/step_158515200.gif" width="260" loading="lazy" alt="step_158515200">
    </td>
    <td align="center"><sub>185M steps</sub><br>
      <img src="videos/step_184934400.gif" width="260" loading="lazy" alt="step_184934400">
    </td>
    <td></td>
  </tr>
</table>

---

## Case Study 2 — Lambda Brax + MJX + A100

MuJoCo XLA compiles the physics to GPU; Brax runs the entire training
loop (env step → rollout → GAE → minibatch SGD) under JIT on-device,
with **4,096 parallel humanoids**. Net throughput ~130,000 steps/sec —
a 37× speedup over the local pipeline.

The headline reward numbers are higher than the local run. The **gait
quality is not**. The policy finds a "speed-skater" exploit: torso
forward, arms stretched behind for balance, one leg as a rudder, front
leg pumping. It looks nothing like walking, but it maximises
forward-velocity reward.

### Lambda v2 — vanilla reward over 766M steps

Peak reward **18,052** at step 766,771,200. Notice how the gait gets
progressively more extreme as the policy leans further into the
reward-hack.

<table>
  <tr>
    <td align="center"><sub>0 (random init)</sub><br>
      <img src="videos_v2/step_000000000.gif" width="260" loading="lazy" alt="v2 step_000000000">
    </td>
    <td align="center"><sub>51M steps</sub><br>
      <img src="videos_v2/step_051118080.gif" width="260" loading="lazy" alt="v2 step_051118080">
    </td>
    <td align="center"><sub>102M steps</sub><br>
      <img src="videos_v2/step_102236160.gif" width="260" loading="lazy" alt="v2 step_102236160">
    </td>
  </tr>
  <tr>
    <td align="center"><sub>153M steps</sub><br>
      <img src="videos_v2/step_153354240.gif" width="260" loading="lazy" alt="v2 step_153354240">
    </td>
    <td align="center"><sub>204M steps</sub><br>
      <img src="videos_v2/step_204472320.gif" width="260" loading="lazy" alt="v2 step_204472320">
    </td>
    <td align="center"><sub>255M steps</sub><br>
      <img src="videos_v2/step_255590400.gif" width="260" loading="lazy" alt="v2 step_255590400">
    </td>
  </tr>
  <tr>
    <td align="center"><sub>306M steps</sub><br>
      <img src="videos_v2/step_306708480.gif" width="260" loading="lazy" alt="v2 step_306708480">
    </td>
    <td align="center"><sub>357M steps</sub><br>
      <img src="videos_v2/step_357826560.gif" width="260" loading="lazy" alt="v2 step_357826560">
    </td>
    <td align="center"><sub>408M steps</sub><br>
      <img src="videos_v2/step_408944640.gif" width="260" loading="lazy" alt="v2 step_408944640">
    </td>
  </tr>
  <tr>
    <td align="center"><sub>460M steps</sub><br>
      <img src="videos_v2/step_460062720.gif" width="260" loading="lazy" alt="v2 step_460062720">
    </td>
    <td align="center"><sub>511M steps</sub><br>
      <img src="videos_v2/step_511180800.gif" width="260" loading="lazy" alt="v2 step_511180800">
    </td>
    <td align="center"><sub>562M steps</sub><br>
      <img src="videos_v2/step_562298880.gif" width="260" loading="lazy" alt="v2 step_562298880">
    </td>
  </tr>
  <tr>
    <td align="center"><sub>613M steps</sub><br>
      <img src="videos_v2/step_613416960.gif" width="260" loading="lazy" alt="v2 step_613416960">
    </td>
    <td align="center"><sub>664M steps</sub><br>
      <img src="videos_v2/step_664535040.gif" width="260" loading="lazy" alt="v2 step_664535040">
    </td>
    <td align="center"><sub>715M steps</sub><br>
      <img src="videos_v2/step_715653120.gif" width="260" loading="lazy" alt="v2 step_715653120">
    </td>
  </tr>
  <tr>
    <td align="center"><sub><strong>766M — peak 18,052</strong></sub><br>
      <img src="videos_v2/step_766771200.gif" width="260" loading="lazy" alt="v2 step_766771200">
    </td>
    <td align="center"><sub>817M (post-NaN recovery)</sub><br>
      <img src="videos_v2/step_817889280.gif" width="260" loading="lazy" alt="v2 step_817889280">
    </td>
    <td></td>
  </tr>
</table>

### Lambda v3 vs v4 — reward shaping vs reward hacking

v3 added an **upright-torso bonus** and a **left/right symmetry
penalty**. v4 stacked on **arm and abdomen penalties** plus a
**forward-lean penalty** to try to kill the speed-skater arms. v4 is
more upright, but the policy still finds new loopholes (e.g. one-leg
skating that technically satisfies left/right symmetry because the
moving leg swaps sides every few frames).

<table>
  <tr>
    <th align="center">Step</th>
    <th align="center">v3 — upright + symmetry</th>
    <th align="center">v4 — + arm / abdomen / lean</th>
  </tr>
  <tr>
    <td align="center"><sub>10M</sub></td>
    <td align="center"><img src="videos_v3/step_010321920.gif" width="260" loading="lazy" alt="v3 10M"></td>
    <td align="center"><img src="videos_v4/step_010321920.gif" width="260" loading="lazy" alt="v4 10M"></td>
  </tr>
  <tr>
    <td align="center"><sub>20M</sub></td>
    <td align="center"><img src="videos_v3/step_020643840.gif" width="260" loading="lazy" alt="v3 20M"></td>
    <td align="center"><img src="videos_v4/step_020643840.gif" width="260" loading="lazy" alt="v4 20M"></td>
  </tr>
  <tr>
    <td align="center"><sub>31M</sub></td>
    <td align="center"><img src="videos_v3/step_030965760.gif" width="260" loading="lazy" alt="v3 31M"></td>
    <td align="center"><img src="videos_v4/step_030965760.gif" width="260" loading="lazy" alt="v4 31M"></td>
  </tr>
  <tr>
    <td align="center"><sub>41M</sub></td>
    <td align="center"><em>(diverged)</em></td>
    <td align="center"><img src="videos_v4/step_041287680.gif" width="260" loading="lazy" alt="v4 41M"></td>
  </tr>
  <tr>
    <td align="center"><sub><strong>52M — peak 18,715 (v4)</strong></sub></td>
    <td align="center"></td>
    <td align="center"><img src="videos_v4/step_051609600.gif" width="260" loading="lazy" alt="v4 52M"></td>
  </tr>
  <tr>
    <td align="center"><sub>62M</sub></td>
    <td align="center"></td>
    <td align="center"><img src="videos_v4/step_061931520.gif" width="260" loading="lazy" alt="v4 62M"></td>
  </tr>
  <tr>
    <td align="center"><sub>72M</sub></td>
    <td align="center"></td>
    <td align="center"><img src="videos_v4/step_072253440.gif" width="260" loading="lazy" alt="v4 72M"></td>
  </tr>
</table>

---

## Reproduce

You'll want two virtual environments — the two pipelines have different
Python/JAX/PyTorch combinations:

1. `.venv/` — Python 3.10, JAX + Brax + MuJoCo-MJX (Lambda GPU training
   + video rendering from MJX checkpoints)
2. `local/.venv/` — Python 3.11, Stable-Baselines3 + PyTorch (local CPU
   training)

### Case study 1 — Local (SB3 + CPU)

```bash
cd local
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fresh training (caffeinate keeps the laptop awake)
caffeinate -i python3 train.py

# Resume from a checkpoint
caffeinate -i python3 train.py --resume checkpoints/<ckpt>.zip

# Record snapshot videos during/after training
python3 snapshots.py
```

Hyperparameters (Zoo-tuned) are in `local/config.yaml`. The full
plateau→fix diff lives in `local/CHANGES.md`.

### Case study 2 — Lambda (Brax + MJX)

1. Copy the env template and fill it in:

    ```bash
    cp .env.example .env
    $EDITOR .env
    ```

    You need:
    - **`LAMBDA_API_KEY`** — [cloud.lambda.ai → API Keys](https://cloud.lambda.ai/api-keys)
    - **`SSH_KEY_NAME`** + **`SSH_KEY_PATH`** — public key added at
      [cloud.lambda.ai → SSH Keys](https://cloud.lambda.ai/ssh-keys),
      matching private key on your machine
    - **`LAMBDA_FILESYSTEM_NAME`** / **`LAMBDA_FILESYSTEM_ID`** /
      **`LAMBDA_FILESYSTEM_MOUNT`** _(optional but strongly
      recommended)_ — a persistent filesystem so your venv +
      checkpoints survive between spot instances. Create one at
      [cloud.lambda.ai → Filesystems](https://cloud.lambda.ai/filesystems).
      Filesystems are regional — `lambda_run.sh` will only launch in
      the filesystem's region when this is set.

2. Local root venv (for rendering + ad-hoc scripting):

    ```bash
    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3. Launch training:

    ```bash
    bash check_availability.sh          # see what GPUs are up right now
    bash lambda_run.sh                  # fresh start
    bash lambda_run.sh --resume         # resume from last FS checkpoint
    ```

    `lambda_run.sh` picks the cheapest available single-GPU instance
    (A10 → A100 → H100 → GH200 → B200 → 8xV100, in that order), uploads
    the code, sets up the venv (reusing a persistent-FS venv if
    available), starts `train.py` under `nohup`, rsyncs checkpoints +
    logs back every 10 minutes, and terminates the instance once
    training finishes.

4. Hyperparameters are in `config.yaml`. Reward function variants
   (vanilla, v3, v4) are in `train.py`.

### Render videos from any MJX checkpoint

```bash
source .venv/bin/activate

# Single checkpoint
python render_mjx.py checkpoints_v4/step_051609600

# Entire run → an output dir
python render_mjx.py checkpoints_v4/ --output videos_v4

# Every 3rd checkpoint, 5 episodes each
python render_mjx.py checkpoints_v2/ --every 3 --episodes 5
```

> **⚠️ Why `render_mjx.py` exists:** MJX checkpoints can't be rendered
> with raw CPU MuJoCo — the `cinert`/`cvel` components of the
> observation differ between MJX and CPU MuJoCo physics, so a policy
> trained under MJX receives garbage inputs and falls over immediately.
> `render_mjx.py` runs the MJX env on CPU JAX (for correct
> observations), then copies `qpos`/`qvel` into CPU MuJoCo (for
> rendering frames).

---

## What didn't work (and why)

- **Default hyperparameters on local SB3** — plateaued at ~650 reward
  for 35M steps. The Optuna-tuned Zoo config broke past it in minutes,
  without any architecture change.
- **Zero entropy coefficient** — no exploration pressure, policy locks
  into local optima. Lifting to 2.4e-3 (SB3) / 1e-3 (Brax) is enough.
- **Unnormalized observations** — Humanoid's sensor values span 5+
  orders of magnitude; PPO can't learn stable representations without
  `VecNormalize` (SB3) / running-stats normalization (Brax).
- **`reward_scaling: 0.1` + no grad clipping (MJX)** — training NaN'd
  in 3 of 4 runs somewhere around 40–80M steps of fine-tuning.
  Dropping to 0.05 and clipping gradients at 1.0 extended runs
  substantially.
- **Hand-crafted gait rewards** — incremental at best. Each time we
  plug one loophole the policy finds another (v3 left/right symmetry
  penalty → v4 one-leg skating that alternates fast enough to satisfy
  symmetry while still being ridiculous). Natural locomotion is
  probably gated on reference-motion imitation (DeepMimic-style).

---

## Further reading

- **[EXPLANATION.md](EXPLANATION.md)** — plain-English walkthrough of
  PPO, reward hacking, and the whole training journey
- **[docs/TECHNICAL.md](docs/TECHNICAL.md)** — full hyperparameters,
  reward functions, architecture, checkpoint formats, pipeline details
- [Stable-Baselines3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) —
  source of the Zoo hyperparameters that broke the plateau
- [Brax](https://github.com/google/brax) — JAX-native RL training loop
- [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) —
  MuJoCo physics compiled to GPU via XLA
- [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html) —
  reference-motion-imitation approach; the likely path to
  actually-natural walking

---

## License

No license file yet — treat this repo as "look but don't ship" until one
is added. Open an issue or PR if you'd like one (MIT is typical for
this kind of writeup).
