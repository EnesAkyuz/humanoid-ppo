# Teaching a Virtual Human to Walk: A Guide

## What is this project?

Imagine you have a virtual ragdoll human in a physics simulator — it has joints, muscles, gravity, and a floor. At the start, it has no idea how to move. It just flops around and falls over. Our goal is to teach it to walk forward, completely on its own, through trial and error.

This is called **reinforcement learning (RL)** — the same idea behind how AlphaGo learned to play Go and how robots learn to do backflips.

## How does a virtual human learn to walk?

### The Setup

Our "human" lives inside **MuJoCo** (Multi-Joint dynamics with Contact), a physics simulator used by robotics researchers worldwide. The humanoid model has:

- **21 motors** (actuators) — one for each joint (hips, knees, ankles, shoulders, elbows, abdomen)
- **336 sensors** — it can "feel" the position and velocity of every body part, plus forces on its joints
- **Gravity, friction, contact physics** — just like the real world

Every 1/60th of a second, the humanoid has to decide: how much force should each of its 21 motors apply? That's 21 numbers it needs to output, and it has to figure out the right combination to walk forward without falling.

### The Reward Signal

We don't tell the humanoid *how* to walk. Instead, we give it a score (reward) based on what happens:

- **+points** for moving forward (the faster, the better)
- **+points** for staying alive (not falling over)
- **-points** for using too much energy (flailing wastes effort)

The humanoid's entire goal is to maximize this score. Over millions of attempts, it figures out that certain movements lead to higher scores than others.

### The Algorithm: PPO

We use an algorithm called **PPO (Proximal Policy Optimization)**. Here's how it works in simple terms:

1. **Try stuff**: The humanoid attempts to walk thousands of times in parallel
2. **See what worked**: We look at which actions led to good outcomes (high reward) and which led to bad outcomes (falling)
3. **Update the brain**: We adjust the humanoid's decision-making (a neural network) to do more of the good stuff and less of the bad stuff
4. **Repeat**: Go back to step 1, but now it's slightly smarter

The "brain" is a **neural network** — a mathematical function with about 200,000 adjustable parameters. At first, these parameters are random, so the humanoid moves randomly. As training progresses, the parameters get tuned so the network outputs increasingly smart movement decisions.

PPO's key trick is that it makes **small, careful updates**. If you change the brain too much at once, it might forget everything it learned. PPO limits how much the policy changes per update, keeping learning stable.

## Our Training Journey

### Attempt 1: The Plateau (Local CPU, 40M steps)

We started with basic settings and trained on a MacBook Pro. The humanoid quickly learned to not fall over immediately (reward ~200), then slowly improved to ~650 reward... and got completely stuck there.

For **35 million more steps** (about 12 hours of training), the reward didn't budge. The humanoid had found a "good enough" strategy — a shuffling stumble — and couldn't discover anything better. This is called a **local optimum**: it's the best option within the small neighborhood of strategies the agent has explored, but there are much better strategies out there it hasn't found.

**What went wrong**: Two things.
1. No observation normalization — the neural network was seeing numbers in wildly different scales (some sensors output 0.001, others output 100+), making learning unstable.
2. No exploration incentive — the entropy coefficient was 0, meaning the agent had no reason to try new things.

### The Fix: Zoo Hyperparameters + VecNormalize

We applied the competition-winning settings from the [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), which were found through automated hyperparameter search (Optuna). The two biggest changes:

1. **VecNormalize**: Automatically scales all observations to have mean=0 and standard deviation=1. This is like converting temperatures from Fahrenheit to a standardized scale — the neural network can learn much more easily when all inputs are on the same scale.

2. **Tuned hyperparameters**: Lower learning rate (so updates are gentler), lower gamma (so the agent focuses more on immediate rewards), and separate neural networks for the actor (movement decisions) and critic (value estimation).

**Result**: The reward shot past the old 650 plateau within minutes, reaching **5,540** at 10M steps and **10,397** peak by 50M steps.

### Going Fast: GPU-Accelerated Training (Lambda A100)

Training on a laptop CPU processes about 3,500 simulation steps per second. We wanted more.

We switched to **MJX** (MuJoCo XLA) — a version of MuJoCo that runs on GPU. Instead of simulating one humanoid at a time, we simulate **4,096 humanoids in parallel** on an NVIDIA A100 GPU.

| Setup | Speed | 50M steps takes |
|-------|-------|-----------------|
| MacBook M4 (CPU) | 3,500 steps/sec | ~4 hours |
| A100 GPU (MJX) | 130,000 steps/sec | ~6 minutes |

That's a **37x speedup**. What took hours now takes minutes, letting us iterate much faster.

**Peak result**: **18,052 reward** at 766M steps — the humanoid running at full speed for the entire episode.

### The Reward Hacking Problem

Here's where it gets funny. Our humanoid did learn to go fast... but not by walking like a human. It discovered an exploit:

- Lean the torso way forward (like a speed skater)
- Stretch both arms behind for balance
- Use one leg as a stick in the back
- Pump the front leg as fast as possible

This "lunge-slide" technique gets insane forward velocity (high reward!) but looks nothing like walking. This is called **reward hacking** — the agent found a loophole in our reward function that we didn't anticipate.

### Reward Shaping: Teaching Posture

To fight reward hacking, we added penalties for the ugly behavior:

- **Upright bonus**: Big reward for keeping the torso vertical, penalty for leaning forward
- **Arm penalty**: Penalize shoulder and elbow joints being far from neutral (arms should hang, not stretch behind)
- **Abdomen penalty**: Penalize the waist from bending (no folding in half)
- **Symmetry cost**: Left and right legs should move similarly (no one-leg hopping)

This improved posture somewhat, but the humanoid is clever — it found new ways to exploit the reward while technically satisfying our constraints. Getting truly natural-looking walking would require **imitation learning** (showing it videos of real human walking), which is a whole different approach.

### The NaN Problem

Three of our four GPU training runs eventually "exploded" — the reward values grew so large that the math overflowed to NaN (Not a Number), corrupting the model. This happened because:

1. As the humanoid gets better, rewards get larger
2. Larger rewards → larger gradients (updates to the neural network)
3. Eventually the gradients get so big they cause numerical overflow

We mitigated this with **gradient clipping** (capping how large updates can be) and **reward scaling** (dividing all rewards by a constant). This helped the training last longer before diverging, giving us more checkpoints to choose from.

## The Numbers

### Local Training (MacBook M4 Pro)

- **Framework**: Stable-Baselines3 (Python, PyTorch)
- **Environment**: 8 parallel humanoids on CPU
- **Training time**: ~4 hours for 50M steps
- **Peak reward**: 10,397
- **Final reward**: ~7,000 (smoothed)
- **Episode length**: ~578 steps (out of 1000 max)

### GPU Training (NVIDIA A100)

- **Framework**: Brax + MJX (Python, JAX)
- **Environment**: 4,096 parallel humanoids on GPU
- **Training time**: ~80 minutes for 766M steps
- **Peak reward**: 18,052
- **Cost**: ~$3 on Lambda Cloud ($1.99/hr)

## What We Learned

1. **Normalization matters more than architecture.** VecNormalize took us from 650 to 5,000+ without changing the neural network at all.

2. **Hyperparameters are everything in RL.** The difference between hand-tuned and Optuna-tuned settings was 10x in performance. Use community-tuned configs as a starting point.

3. **GPU acceleration is a game-changer for iteration speed.** Being able to test an idea in 6 minutes vs 4 hours means you can try 40x more experiments in the same time.

4. **Agents will exploit your reward function.** If there's a loophole, they'll find it. Reward design is an art — you need to think adversarially about what behaviors your reward function accidentally encourages.

5. **Save checkpoints frequently.** Training can diverge unpredictably. Without checkpoints, hours of training can be lost. With them, you just pick the best one.

6. **Natural locomotion is hard.** Getting a humanoid to move fast is relatively easy. Getting it to move fast *while looking like a human* is an open research problem. The best approaches use motion capture data to teach the agent to imitate real human movement.

## Glossary

- **PPO (Proximal Policy Optimization)**: An algorithm that learns by trying things and updating its strategy in small, stable steps
- **Neural Network**: A mathematical function with tunable parameters that maps observations to actions
- **Reward**: A numerical score telling the agent how well it's doing
- **Episode**: One attempt at the task, from start until the humanoid falls (or reaches the time limit)
- **Timestep**: One moment of decision — the agent observes, acts, and gets a reward
- **VecNormalize**: A wrapper that automatically scales observations to a standard range
- **MuJoCo**: The physics engine simulating the humanoid's body and environment
- **MJX**: MuJoCo compiled to run on GPU via JAX
- **Reward hacking**: When an agent finds an unintended way to get high reward without solving the task as intended
- **NaN divergence**: When numerical values overflow to "Not a Number," breaking the training
- **Checkpoint**: A saved snapshot of the agent's brain at a point in training
- **Local optimum**: A solution that's better than nearby alternatives but not the best overall — like being on top of a small hill while there's a mountain nearby you can't see
