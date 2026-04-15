# Training Changes Log

## 2026-04-06 — Breaking the Plateau

Training plateaued at ~650 mean reward / ~125 episode length from ~5M to 40M steps. The humanoid was doing a basic shuffle but not properly walking.

### Changes

| Setting | Before | After | Reason |
|---------|--------|-------|--------|
| `n_envs` | 4 | 8 | More diverse experience per update |
| `learning_rate` | 3e-4 | 1e-4 | Finer optimization past the plateau |
| `ent_coef` | 0.0 | 0.005 | Add exploration incentive to escape local optimum |
| `net_arch` | [256, 256] | [512, 256, 128] | Wider network (fresh start only, can't change on resume) |

### Why these changes

- **Zero entropy** meant no exploration pressure — the policy found a "good enough" local optimum and stayed there.
- **Lower LR** lets the optimizer take finer steps instead of bouncing around the same region.
- **More envs** give more diverse rollouts per update, helping the policy see a wider range of states.
- **Bigger network** gives more capacity to represent a complex walking gait (only applies on fresh training, architecture is baked into saved checkpoints).

### Resume command

```bash
source .venv/bin/activate
caffeinate -i python3 train.py --resume checkpoints/humanoid_ppo_40000000_steps.zip
```

`train.py` now applies the new hyperparams (lr, ent_coef, etc.) to the loaded model on resume.
