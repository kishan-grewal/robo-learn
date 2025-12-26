# sac_pendulum/sweep.py

import optuna
import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim

from train import Actor, TwinCritic, ReplayBuffer, state_to_tensor

HIDDEN_LAYER_NODES_ACTOR = 256
HIDDEN_LAYER_NODES_CRITIC = 256
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class Config:
    # Training
    total_timesteps: int = 35000
    buffer_size: int = 100000
    min_buffer_train: int = 1000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha: float = 0.2

    # Optuna
    n_trials: int = 4
    prune_interval: int = 10000
    prune_lookback: int = 10
    final_lookback: int = 20
    failed_reward: float = -1600.0
    top_k: int = 5


def train(trial, config=Config()):
    # Optuna suggests hyperparameters
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])

    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim, hidden_size, LOG_STD_MIN, LOG_STD_MAX)
    critic = TwinCritic(state_dim, action_dim, hidden_size)
    target_critic = TwinCritic(state_dim, action_dim, hidden_size)
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=config.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.lr)

    buffer = ReplayBuffer(config.buffer_size)

    total_reward_array = []
    obs, _ = env.reset()
    ep_return = 0.0

    for step in range(config.total_timesteps):
        if step < config.min_buffer_train:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _ = actor.sample(state_to_tensor(obs))
                action = action.squeeze(0).numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.store(obs, action, reward, next_obs, done)
        ep_return += reward
        obs = next_obs

        if done:
            total_reward_array.append(ep_return)
            ep_return = 0.0
            obs, _ = env.reset()

        if step >= config.min_buffer_train:
            states, actions, rewards, next_states, dones = buffer.sample(
                config.batch_size
            )

            # Critic update
            with torch.no_grad():
                next_actions, next_log_probs = actor.sample(next_states)
                q1_target, q2_target = target_critic(next_states, next_actions)
                q_target = torch.min(q1_target, q2_target)
                target = (
                    rewards
                    + config.gamma
                    * (q_target - config.alpha * next_log_probs)
                    * (~dones).float()
                )

            q1, q2 = critic(states, actions)
            critic_loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor update
            actions_, log_probs = actor.sample(states)
            q1_, q2_ = critic(states, actions_)
            q_ = torch.min(q1_, q2_)
            actor_loss = (config.alpha * log_probs - q_).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Soft target update
            with torch.no_grad():
                for param, target_param in zip(
                    critic.parameters(), target_critic.parameters()
                ):
                    target_param.data.copy_(
                        (1 - config.tau) * target_param.data + config.tau * param.data
                    )

        # Optuna pruning
        if (
            step % config.prune_interval == 0
            and step > 0
            and len(total_reward_array) > 0
        ):
            trial.report(np.mean(total_reward_array[-config.prune_lookback :]), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    env.close()

    # Return metric to optimize
    if len(total_reward_array) >= config.final_lookback:
        return np.mean(total_reward_array[-config.final_lookback :])
    return config.failed_reward


if __name__ == "__main__":
    config = Config()

    study = optuna.create_study(direction="maximize")
    study.optimize(train, n_trials=config.n_trials, show_progress_bar=True)

    # Results (filter out failed/pruned trials)
    print(f"\nBest reward: {study.best_trial.value:.1f}")
    print(f"Best params: {study.best_trial.params}")

    completed_trials = [t for t in study.trials if t.value is not None]
    trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)

    print(f"\nTop {config.top_k} trials:")
    for i, t in enumerate(trials[: config.top_k]):
        print(f"{i+1}. Reward: {t.value:.1f} | {t.params}")
