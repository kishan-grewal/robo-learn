# ppo_gae_pendulum/train.py

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import gymnasium as gym
import os

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_LAYER_NODES_ACTOR = 64
HIDDEN_LAYER_NODES_CRITIC = 64
ACTION_HIGH = 2.0
LOG_STD_MIN = -2.0
LOG_STD_MAX = 1.0


# DATA CLASSES
class Config:
    total_timesteps: int = 200000
    rollout_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon: float = 0.2
    lr: float = 3e-4


# HELPER FUNCTIONS
def state_to_tensor(state):
    if isinstance(state, np.ndarray):
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    return torch.tensor([state], dtype=torch.float32)


# CLASSES
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.action_high = ACTION_HIGH
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX

        self.actor_body = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_LAYER_NODES_ACTOR),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYER_NODES_ACTOR, HIDDEN_LAYER_NODES_ACTOR),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(HIDDEN_LAYER_NODES_ACTOR, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_LAYER_NODES_CRITIC),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYER_NODES_CRITIC, HIDDEN_LAYER_NODES_CRITIC),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYER_NODES_CRITIC, 1),
        )

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def _get_dist(self, obs):
        h = self.actor_body(obs)
        mu = self.mu_head(h)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return torch.distributions.Normal(mu, std)

    def get_action_and_logprob(self, obs):
        dist = self._get_dist(obs)

        z = dist.rsample()
        a = torch.tanh(z)
        action_env = a * self.action_high

        logprob = dist.log_prob(z).sum(dim=-1)
        logprob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return action_env, logprob, entropy, z

    def logprob_of_action(self, obs, action_env):
        dist = self._get_dist(obs)

        a = torch.clamp(action_env / self.action_high, -0.999, 0.999)
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))

        logprob = dist.log_prob(z).sum(dim=-1)
        logprob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logprob, entropy


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    config = Config()
    actor_critic = ActorCritic(state_dim, action_dim)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=config.lr)

    # Pre-allocated tensor buffers (instead of Python lists)
    obs_buf = torch.zeros((config.rollout_steps, state_dim))
    act_buf = torch.zeros((config.rollout_steps, action_dim))
    rew_buf = torch.zeros(config.rollout_steps)
    done_buf = torch.zeros(config.rollout_steps)
    logprob_buf = torch.zeros(config.rollout_steps)
    value_buf = torch.zeros(config.rollout_steps)

    # plot reward
    total_reward_array = []

    # global step and num updates
    global_step = 0
    num_updates = config.total_timesteps // config.rollout_steps

    obs, _ = env.reset()
    ep_return = 0.0

    for update in trange(num_updates):
        # ROLLOUT COLLECTION
        for step in range(config.rollout_steps):
            global_step += 1

            with torch.no_grad():
                obs_tensor = state_to_tensor(obs)
                action_env, logprob, entropy, z = actor_critic.get_action_and_logprob(
                    obs_tensor
                )
                value = actor_critic.value(obs_tensor)
                action_np = action_env.squeeze().numpy()

            next_obs, reward, terminated, truncated, info = env.step([action_np])
            done = terminated or truncated

            # Store in buffers
            obs_buf[step] = obs_tensor.squeeze(0)
            act_buf[step] = action_env.squeeze(0)
            rew_buf[step] = reward
            done_buf[step] = 1.0 if done else 0.0
            logprob_buf[step] = logprob.squeeze(0)
            value_buf[step] = value.squeeze(0)

            ep_return += reward
            obs = next_obs

            if done:
                total_reward_array.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()

        # GAE COMPUTATION (with proper done handling)
        with torch.no_grad():
            next_value = actor_critic.value(state_to_tensor(obs)).squeeze()

        advantages = torch.zeros(config.rollout_steps)
        gae = 0.0

        for t in reversed(range(config.rollout_steps)):
            if t == config.rollout_steps - 1:
                next_nonterminal = 1.0 - done_buf[t]
                next_val = next_value
            else:
                next_nonterminal = 1.0 - done_buf[t + 1]
                next_val = value_buf[t + 1]

            delta = (
                rew_buf[t] + config.gamma * next_val * next_nonterminal - value_buf[t]
            )
            gae = delta + config.gamma * config.gae_lambda * next_nonterminal * gae
            advantages[t] = gae

        returns = advantages + value_buf

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # MINIBATCH TRAINING (batched, not per-sample)
        for epoch in range(config.update_epochs):
            indices = torch.randperm(config.rollout_steps)

            for start in range(0, config.rollout_steps, config.minibatch_size):
                mb_indices = indices[start : start + config.minibatch_size]

                # Gather minibatch (all at once)
                mb_obs = obs_buf[mb_indices]
                mb_act = act_buf[mb_indices]
                mb_logprob_old = logprob_buf[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Forward pass on entire minibatch
                logprob_new, _ = actor_critic.logprob_of_action(mb_obs, mb_act)
                values_new = actor_critic.value(mb_obs)

                # Policy loss
                ratio = torch.exp(logprob_new - mb_logprob_old)
                surrogate = ratio * mb_advantages
                clipped_surrogate = (
                    torch.clamp(ratio, 1 - config.epsilon, 1 + config.epsilon)
                    * mb_advantages
                )
                policy_loss = -torch.min(surrogate, clipped_surrogate).mean()

                # Value loss
                value_loss = ((mb_returns - values_new) ** 2).mean()

                # Total loss and update
                total_loss = policy_loss + value_loss

                ac_optimizer.zero_grad()
                total_loss.backward()
                ac_optimizer.step()

    # PLOT TOTAL REWARD
    total_rewards = np.array(total_reward_array)
    window = 100
    total_reward_moving_avg = np.convolve(
        total_rewards, np.ones(window) / window, mode="valid"
    )

    plt.plot(total_rewards, alpha=0.3, label="Training")
    plt.plot(
        range(window - 1, len(total_rewards)),
        total_reward_moving_avg,
        label=f"{window}-episode moving average",
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward")
    plt.legend()
    plt.show()

    # save the model
    os.makedirs("models_pendulum", exist_ok=True)
    torch.save(actor_critic.state_dict(), "models_pendulum/pendulum_ppo_gae_ac.pth")
    print("Model saved to models_pendulum/pendulum_ppo_gae_ac.pth")
