# dqn_1d/train.py

# IMPORTS
import random
from collections import deque  # replay buffer

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange


# DATA CLASSES
class Config:
    epsilon: float = 0.3
    gamma: float = 0.9
    lr: float = 1e-3


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


# used to give us a history of some of the transitions so we can train without fitting to time
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def store(self, state, action, reward, next_state, done):
        transition = (
            state,
            action,
            reward,
            next_state,
            done,
        )  # just a tuple, we add one at a time with .store
        self.buffer.append(transition)

    def sample(self, batch_size):
        # this is like going from looking at each row of the batch matrix [s1 a1 r1 .. s2 a2 r2..] to looking at each column [s1 s2 s3 .. a1 a2 a3]
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # weird pytorch stuff
        states = torch.tensor(states, dtype=torch.float32).unsqueeze(
            1
        )  # used by the network, network wants (batch_size, 1) not (batch_size,)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32).unsqueeze(
            1
        )  # used by the network too
        dones = torch.tensor(dones, dtype=torch.bool)
        # we unsqueeze when a tensor needs to match a spatial/feature dimension, when feeding into a layer,
        # all other times it is just data values that participate in math operations, and we should not unsqueeze
        # its like an explicit matrix of vector size, vs just an actual vector
        # a layer cannot take in a vector, but we need the vector for math (x, 1) vs (x,)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def is_full(self):
        return len(self) == self.capacity


if __name__ == "__main__":
    config = Config()

    # SETUP
    env = ValueWorld()
    qnet = QNetwork()
    optimizer = optim.Adam(qnet.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    epsilon = config.epsilon  # exploration rate
    gamma = config.gamma  # discount factor

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
