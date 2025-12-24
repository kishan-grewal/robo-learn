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

HIDDEN_LAYER_NODES = 32


# DATA CLASSES
class Config:
    episode_count: int = 650
    gamma: float = 0.99
    lr: float = 1e-3
    # replay buffer
    buffer_capacity: int = 5000
    buffer_mintrain: int = 500
    batch_size: int = 64
    # epsilon decay
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    # target network
    target_update_freq: int = 100


# HELPER FUNCTIONS
def state_to_tensor(state):
    # state is shape (4,) from gym
    # Need to convert to (1, 4) for network
    if isinstance(state, np.ndarray):
        return torch.tensor(state, dtype=torch.float32).unsqueeze(
            0
        )  # add dim 0 (mat to vector)
    return torch.tensor([state], dtype=torch.float32)


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
            nn.Linear(state_dim, HIDDEN_LAYER_NODES),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_NODES, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def store(self, obs, action, reward, next_obs, done):
        transition = (
            obs,
            action,
            reward,
            next_obs,
            done,
        )
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obss, actions, rewards, next_obss, dones = zip(*batch)

        # obss and next_obss is already (b, 4) so no unsqueeze
        obss = torch.from_numpy(np.array(obss)).float()
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_obss = torch.from_numpy(np.array(next_obss)).float()
        dones = torch.tensor(dones, dtype=torch.bool)

        return obss, actions, rewards, next_obss, dones

    def __len__(self):
        return len(self.buffer)

    def is_full(self):
        return len(self) == self.capacity


if __name__ == "__main__":
    print("observation = incomplete state, the state is the true real world state")

    # setup env
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2

    # setup qnet
    config = Config()
    qnet = QNetwork(state_dim, action_dim)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(qnet.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_capacity)
    target_qnet = QNetwork(state_dim, action_dim)

    # plot reward
    reward_array = []

    # plot loss
    loss_array = []

    # initialise epsilon
    epsilon = config.epsilon_start

    # initialise target network
    target_qnet.load_state_dict(qnet.state_dict())
    step_for_target_update = 0

    # episode loop
    for episode in trange(config.episode_count):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            # increment step
            step_for_target_update += 1

            # select action
            action = select_action(qnet, obs, epsilon)
            epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

            # take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # learn 2
            buffer.store(obs, action, reward, next_obs, done)

            if len(buffer) >= config.buffer_mintrain:
                obss, actions, rewards, next_obss, dones = buffer.sample(
                    config.batch_size
                )
                q_values = qnet(obss)
                # filter by what actions we actually chose (actions index the q value matrix)
                q_value = q_values.gather(1, actions.unsqueeze(1))
                # (b, 1) -> (b,) which is matrix to vector
                q_value = q_value.squeeze(1)

                with torch.no_grad():
                    next_q_values = target_qnet(next_obss)
                    # 0 for max 1 for argmax, and we want the max in each row (max action for each state)
                    max_next_q_values = next_q_values.max(dim=1)[0]
                    target = (
                        rewards + config.gamma * max_next_q_values * (~dones).float()
                    )

                loss = loss_fn(q_value, target)
                loss_array.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step_for_target_update % config.target_update_freq == 0:
                target_qnet.load_state_dict(qnet.state_dict())

            # # learn 1
            # q_values = qnet(state_to_tensor(obs))

            # q_value = q_values[0, action]

            # with torch.no_grad():
            #     next_q_values = qnet(state_to_tensor(next_obs))
            #     target = reward + config.gamma * next_q_values.max()

            # reward
            total_reward += reward
            obs = next_obs

        reward_array.append(total_reward)

    # PLOT THE REWARD
    rewards = np.array(reward_array)
    window = 50
    reward_moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.plot(rewards, alpha=0.3, label="Training")
    plt.plot(
        range(window - 1, len(rewards)),
        reward_moving_avg,
        label=f"{window}-episode moving average",
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("CartPole Training")
    plt.legend()
    plt.show()

    # PLOT THE LOSS
    losses = np.array(loss_array)
    window = 25
    loss_moving_avg = np.convolve(losses, np.ones(window) / window, mode="valid")

    plt.plot(losses, alpha=0.3, label="Loss")
    plt.plot(
        range(window - 1, len(losses)),
        loss_moving_avg,
        label=f"{window}-episode moving average",
    )
    plt.xlabel("Episode")
    plt.ylabel("MSE")
    plt.title("CartPole Training")
    plt.legend()
    plt.show()
