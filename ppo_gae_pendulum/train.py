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
    episode_count: int = 800
    epoch_count: int = 3
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

    # plot reward
    total_reward_array = []

    # episode loop
    for episode in trange(config.episode_count):
        obs, info = env.reset()
        done = False

        episode_states = []
        episode_actions = []
        episode_rewards = []
        # for clipped surrogate loss
        episode_log_probs_old = []
        episode_values = []

        while not done:
            with torch.no_grad():
                action_env, logprob, entropy, z = actor_critic.get_action_and_logprob(state_to_tensor(obs))
                value = actor_critic.value(state_to_tensor(obs))
                action_np = action_env.squeeze().numpy()

            next_obs, reward, terminated, truncated, info = env.step([action_np])
            done = terminated or truncated

            episode_states.append(obs)
            episode_actions.append(action_env)
            episode_rewards.append(reward)
            episode_log_probs_old.append(logprob)
            episode_values.append(value)

            obs = next_obs

        total_reward_array.append(sum(episode_rewards))

        # GAE COMPUTATION
        rewards = torch.tensor(episode_rewards, dtype=torch.float32)
        values = torch.stack(episode_values)
        
        # Bootstrap: get value of next state (or 0 if done)
        next_obs_tensor = state_to_tensor(obs)
        with torch.no_grad():
            next_value = actor_critic.value(next_obs_tensor)
        
        # Compute TD residuals (deltas)
        deltas = rewards + config.gamma * next_value - values
        
        # Compute GAE advantages
        advantages = torch.zeros_like(deltas)
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + config.gamma * config.gae_lambda * gae
            advantages[t] = gae
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # TRAINING LOOP
        for epoch in range(config.epoch_count):
            ac_optimizer.zero_grad()
            
            total_policy_loss = 0
            total_value_loss = 0
            
            for t, (obs, action) in enumerate(zip(episode_states, episode_actions)):
                logprob_new, entropy = actor_critic.logprob_of_action(state_to_tensor(obs), action)
                logprob_old = episode_log_probs_old[t].detach()
                
                ratio = torch.exp(logprob_new - logprob_old)
                surrogate = ratio * advantages[t]
                clipped_surrogate = torch.clamp(ratio, 1 - config.epsilon, 1 + config.epsilon) * advantages[t]
                policy_loss = -torch.min(surrogate, clipped_surrogate)
                
                value_pred = actor_critic.value(state_to_tensor(obs))
                value_loss = (returns[t] - value_pred) ** 2
                
                total_policy_loss += policy_loss
                total_value_loss += value_loss

            total_loss = (total_policy_loss + total_value_loss).mean()
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