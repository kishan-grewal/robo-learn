# gym-rl-algos

Implementations of common reinforcement learning algorithms for Gymnasium environments. Built for understanding how these algorithms actually work in practice.

![SAC HalfCheetah](media/sac_cheetah_10secs.gif)

## Algorithms

| Algorithm | Environment | Type | Key Concepts |
|-----------|-------------|------|--------------|
| DQN | Custom 1D, CartPole | Discrete, Off-policy | Replay buffer, target networks, epsilon-greedy |
| Policy Gradient | CartPole | Discrete, On-policy | REINFORCE, value baseline |
| PPO | CartPole, Pendulum | Discrete/Continuous, On-policy | Clipped surrogate, advantage normalisation |
| PPO + GAE | Pendulum, HalfCheetah | Continuous, On-policy | Generalised Advantage Estimation, minibatch updates |
| PPO Vectorised | HalfCheetah | Continuous, On-policy | Parallel environments, 2M timestep training |
| SAC | Pendulum, HalfCheetah | Continuous, Off-policy | Entropy regularisation, twin critics, learned alpha |
| MBPO | Pendulum, HalfCheetah | Continuous, Model-based | Dynamics ensemble, imaginary rollouts |

## Structure

```
robo-learn/
├── dqn_1d/              # Simple 3-state environment for learning DQN basics
├── dqn_cartpole/        # DQN with Double DQN on CartPole-v1
├── pg_cartpole/         # Vanilla policy gradient with value baseline
├── ppo_cartpole/        # PPO with clipped surrogate on discrete actions
├── ppo_pendulum/        # PPO for continuous control (tanh squashing)
├── ppo_gae_pendulum/    # PPO + GAE with proper done handling
├── ppo_vec_cheetah/     # Vectorised PPO for HalfCheetah (16 parallel envs)
├── sac_pendulum/        # SAC with twin critics
├── sac_cheetah/         # SAC on HalfCheetah-v5
├── mbpo_pendulum/       # Model-Based Policy Optimisation
├── mbpo_cheetah/        # MBPO with probabilistic dynamics ensemble
└── envs/                # Conda environment files
```

Each directory contains:
- `train.py` - Training script with full implementation
- `evaluate.py` - Load and run trained models
- `print.py` - Environment inspection is present in the first variant of each gym task

## Setup

```bash
# GPU
conda env create -f envs/rl-gpu.yml
conda activate rl-gpu
pip install -r envs/requirements.txt

# CPU
conda env create -f envs/rl-cpu.yml
conda activate rl-cpu
pip install -r envs/requirements.txt
```

## Noteworthy Results

**PPO on HalfCheetah** (vectorised, 2M steps): ~2000 reward  
**SAC on HalfCheetah** (300k steps): ~4000 reward  

PPO needs some tuning as is evident from the low reward of 2000 and noisy training.

## Implementation Notes

**Continuous actions**: Gaussian policies with tanh squashing. The log-prob correction `logprob -= log(1 - tanh(z)²)` is required for correct gradients.

**GAE**: Accumulates TD errors backwards with λ-discounting. Separate from γ because λ controls the bias-variance tradeoff of the *estimator*, not the objective.

**Vectorised environments**: `gym.vector.SyncVectorEnv` runs N environments in parallel. Buffers are shaped `(rollouts_per_env, num_envs, ...)` so time indexing is straightforward.

**MBPO**: Trains a dynamics ensemble on real data, generates synthetic rollouts, mixes real and imaginary data for SAC updates. Short rollouts (1-5 steps) avoid compounding model error.