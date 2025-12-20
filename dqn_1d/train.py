# dqn_1d/train.py

# IMPORTS
import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange


# HELPER FUNCTIONS
def state_to_tensor(state):
    return torch.tensor([[state]], dtype=torch.float32)


def select_action(qnet, state, epsilon):
    """p(choose random action) = epsilon else highest Q-value"""
    if random.random() < epsilon:
        return random.randint(0, 2)

    with torch.no_grad():
        q_values = qnet(state_to_tensor(state))
        return q_values.argmax().item()


# CLASSES
class ValueWorld:
    def __init__(self):
        self.reset()

    def reset(self):
        self.position = 0
        self.steps = 0
        return self.position

    def step(self, action):
        if action == 0:
            self.position = max(0, self.position - 1)  # left
        elif action == 2:
            self.position = min(2, self.position + 1)  # right
        # action == 1 -> stay so no if clause

        reward = 0.0
        done = False

        # danger
        if self.position == 1:
            reward = -1.0
            if random.random() < 0.3:
                reward = -5.0
                done = True

        # goal
        if self.position == 2:
            reward = 10.0
            done = True

        return self.position, reward, done


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # SETUP
    env = ValueWorld()
    qnet = QNetwork()
    optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epsilon = 0.3  # exploration rate
    gamma = 0.9  # discount factor

    # EPISODE LOOP
    for episode in range(10):
        state = env.reset()
        total_reward = 0
        done = False

        print(f"Starting episode {episode}")

        # STEP LOOP
        for t in range(5):
            # STEP
            action = select_action(qnet, state, epsilon)

            next_state, reward, done = env.step(action)

            print(
                f"t={t} | state={state} | action={action} | reward={reward} | next_state={next_state}"
            )

            # LEARN
            q_values = qnet(state_to_tensor(state))

            q_value = q_values[0, action]

            with torch.no_grad():
                next_q_values = qnet(state_to_tensor(next_state))
                target = reward + gamma * next_q_values.max()

            loss = loss_fn(q_value, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # REWARD
            total_reward += reward
            state = next_state

            if done:
                break

        print("Episode finished")
        print("Total reward:", total_reward)
