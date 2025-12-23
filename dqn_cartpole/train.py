# dqn_cartpole/train.py

# IMPORTS
import random
from collections import deque  # replay buffer

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange

import gymnasium as gym

HIDDEN_LAYER_Nodes = 32


# DATA CLASSES
class Config:
    episode_count: int = 1000
    gamma: float = 0.99
    lr: float = 1e-3
    epsilon: float = 1.0  # add decay later


# HELPER FUNCTIONS
def state_to_tensor(state):
    return torch.tensor([[state]], dtype=torch.float32)


def select_action(qnet, state, epsilon):
    """epsilon greedy"""
    if random.random() < epsilon:
        return random.randint(0, 1)

    with torch.no_grad():
        q_values = qnet(state_to_tensor(state))
        return q_values.argmax().item()


# CLASSES
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_LAYER_Nodes),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_Nodes, action_dim),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    print("observation = incomplete state, the state is the true real world state")

    # setup env
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2

    # setup qnet
    config = Config()
    qnet = QNetwork(state_dim, action_dim)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(qnet.parameters(), lr=config.lr)

    # plot reward
    reward_array = []

    # episode loop
    for episode in trange(config.episode_count):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            # select action
            action = select_action(qnet, obs, config.epsilon)

            # take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # learn

            # reward
            total_reward += reward
            obs = next_obs

        reward_array.append(total_reward)

    # all plots
    rewards = np.array(reward_array)
    window = 100
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.plot(rewards, alpha=0.3, label="Training")
    plt.plot(
        range(window - 1, len(rewards)),
        moving_avg,
        label=f"{window}-episode moving average",
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("CartPole Training")
    plt.legend()
    plt.show()
