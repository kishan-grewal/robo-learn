# robo-learn

To begin with, install **miniconda** and run the first **OR** second set of commands depending on whether you have a GPU

### GPU
```bash
# Create GPU environment (PyTorch + CUDA)
conda env create -f envs/rl-gpu.yml
conda config --set channel_priority strict
conda activate rl-gpu

# Install Python-level dependencies
pip install -r envs/requirements.txt
```

### CPU
```bash
# Create CPU-only environment (PyTorch + cpuonly)
conda env create -f envs/rl-cpu.yml
conda config --set channel_priority strict
conda activate rl-cpu

# Install Python-level dependencies
pip install -r envs/requirements.txt
```

#### Key Insights

DQN: discrete, off-policy
PPO: discrete or continuous, on-policy
SAC: continuous, off-policy

Replay buffer (off-policy) and minibatching (on-policy) usually give the most boost to training

Followed by correct algorithms (no bugs)

#### Also learnt:

- Value baselines reduce variance in policy gradients
- Advantage normalization stabilizes training
- Per-timestep returns (not episode total) for proper credit assignment
- Batched updates (once per episode/rollout) vs per-timestep updates
- Continuous actions need Gaussian policies (μ, σ) instead of softmax
- Tanh squashing maps unbounded Gaussian samples to bounded actions [-2, 2]
- Log-prob correction needed when using tanh: `logprob -= log(1 - tanh(z)²)`
- Done handling in GAE - don't bootstrap across episode boundaries
- Larger rollouts (2048 steps) give more stable gradients than single episodes (200 steps)
```

---

**2. On-policy vs off-policy:**

**On-policy:** Learn from data you *just* collected with your *current* policy. Use it once, throw it away.
```
collect with π_current → train π_current → data is now stale → collect again
```

PPO does this. The ratio `π_new / π_old` only works if `π_old` is very recent.

**Off-policy:** Learn from data collected by *any* policy, including old versions of yourself.
```
collect with π_old → store in buffer → sample randomly → train π_current
