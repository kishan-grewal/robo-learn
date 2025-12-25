# ppo_cartpole/train.py

# IMPORTS
import random

# no replay buffer for on-policy PG

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange

import gymnasium as gym

import os

HIDDEN_LAYER_NODES_POLICY = 64
HIDDEN_LAYER_NODES_VALUE = 64


# DATA CLASSES
class Config:
    episode_count: int = 500
    epoch_count: int = 3  # standard for PPO
    gamma: float = 0.99
    epsilon: float = 0.2  # standard for PPO 1 +- eps
    lr: float = 1e-3


# HELPER FUNCTIONS
def state_to_tensor(state):
    if isinstance(state, np.ndarray):
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    return torch.tensor([state], dtype=torch.float32)


# CLASSES
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_LAYER_NODES_POLICY),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_NODES_POLICY, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_LAYER_NODES_VALUE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_NODES_VALUE, 1),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    config = Config()
    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)
    # dont need loss_fn, we compute with a different formula to MSE
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=config.lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=config.lr)

    # plot reward
    total_reward_array = []

    # plot both losses
    policy_loss_array = []
    value_loss_array = []

    # episode loop
    for episode in trange(config.episode_count):
        obs, info = env.reset()
        done = False

        episode_states = []
        episode_actions = []
        episode_rewards = []
        # clipped surrogate loss for ppo
        episode_log_probs_old = []

        while not done:
            with torch.no_grad():
                logits = policy_net(state_to_tensor(obs))
                # logits.shape: (1, 4)
                probs = torch.softmax(logits, dim=1)
                # probs.shape: (1, 2)
                action = torch.multinomial(probs, 1).item()
                # action: int

                # FOR USE IN CLIPPED SURROGATE LOSS FOR PPO
                # log_softmax(logits, dim=1).shape: (1, 2)
                log_prob_old = torch.log_softmax(logits, dim=1)[0, action]

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_states.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            # clipped surrogate loss for ppo
            episode_log_probs_old.append(log_prob_old)

            obs = next_obs

        total_reward_array.append(sum(episode_rewards))
        # still need to discount with gamma
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + config.gamma * G
            returns.insert(0, G)  # adds to it from the back to front

        # train on collected episode !!!
        for epoch in range(config.epoch_count):
            # tensor creation
            returns = torch.tensor(returns, dtype=torch.float32)
            baselines = torch.cat(
                [value_net(state_to_tensor(obs)) for obs in episode_states]
            ).squeeze()
            advantages = returns - baselines

            # normalize advantages (FOR PPO)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # detach advantages from the gradient graph, it isnt used for gradient descent
            # cannot be backpropped through
            advantages = advantages.detach()

            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()

            total_policy_loss = 0
            total_value_loss = 0

            for t, (obs, action) in enumerate(zip(episode_states, episode_actions)):
                G_t = returns[t]
                baseline_t = baselines[t]
                advantage_t = advantages[t]

                logits = policy_net(state_to_tensor(obs))
                log_probs = torch.log_softmax(logits, dim=1)
                log_prob = log_probs[0, action]

                # surrogate = (π_new / π_old) * advantage
                # surrogate = (p_new / p_old) * (G - v(s))
                log_prob_old = episode_log_probs_old[t].detach()
                ratio = torch.exp(log_prob - log_prob_old)
                surrogate = ratio * advantage_t
                # clip the surrogate
                clipped_surrogate = (
                    torch.clamp(ratio, 1 - config.epsilon, 1 + config.epsilon)
                    * advantage_t
                )
                policy_loss = -torch.min(surrogate, clipped_surrogate)

                total_policy_loss += policy_loss
                total_value_loss += (G_t - baseline_t) ** 2

            # Single update per epoch
            total_policy_loss.backward()
            policy_optimizer.step()

            total_value_loss.backward()
            value_optimizer.step()

            policy_loss_array.append(total_policy_loss.item())
            value_loss_array.append(total_value_loss.item())

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

    # PLOT POLICY LOSS
    policy_losses = np.array(policy_loss_array)
    window = 100
    policy_loss_moving_avg = np.convolve(
        policy_losses, np.ones(window) / window, mode="valid"
    )

    plt.plot(policy_losses, alpha=0.3, label="Loss")
    plt.plot(
        range(window - 1, len(policy_losses)),
        policy_loss_moving_avg,
        label=f"{window}-episode moving average",
    )
    plt.xlabel("Episode")
    plt.ylabel("Loss -(G-v(s))logp)")
    plt.title("Policy")
    plt.legend()
    plt.show()

    # PLOT VALUE LOSS
    value_losses = np.array(value_loss_array)
    window = 100
    value_loss_moving_avg = np.convolve(
        value_losses, np.ones(window) / window, mode="valid"
    )

    plt.plot(value_losses, alpha=0.3, label="Loss")
    plt.plot(
        range(window - 1, len(value_losses)),
        value_loss_moving_avg,
        label=f"{window}-episode moving average",
    )
    plt.xlabel("Episode")
    plt.ylabel("Loss (G-v(s))^2")
    plt.title("Value")
    plt.legend()
    plt.show()

    # save the model
    os.makedirs("models", exist_ok=True)
    torch.save(policy_net.state_dict(), "models/cartpole_ppo_policy.pth")
    print("Model saved to models/cartpole_ppo_policy.pth")
    torch.save(value_net.state_dict(), "models/cartpole_ppo_value.pth")
    print("Model saved to models/cartpole_ppo_value.pth")
