# dqn_1d/train.py

# IMPORTS
import random
from collections import deque  # replay buffer

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange

HIDDEN_LAYER_NODES = 32


# DATA CLASSES
class Config:
    episode_count: int = 1000
    step_count: int = 20
    gamma: float = 0.9
    lr: float = 1e-3
    buffer_max_capacity: int = 1000
    buffer_min_train: int = 32
    batch_size: int = 32
    target_update_freq: int = 50
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995


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
        self.net = nn.Sequential(
            nn.Linear(1, HIDDEN_LAYER_NODES),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_NODES, 3),
        )

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
    # SETUP
    env = ValueWorld()
    qnet = QNetwork()
    loss_fn = nn.MSELoss()

    config = Config()
    optimizer = optim.Adam(qnet.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_max_capacity)
    target_qnet = QNetwork()

    # PLOT reward
    reward_array = []

    # MSE error
    loss_array = []

    # target network and step count (train on a stable target more than 1 step)
    target_qnet.load_state_dict(qnet.state_dict())
    step_counter = 0

    # epsilon greedy parameter decay (become less explorative)
    epsilon = config.epsilon_start

    # EPISODE LOOP
    for episode in trange(config.episode_count):
        state = env.reset()
        total_reward = 0
        done = False

        print(f"Starting episode {episode}")

        # STEP LOOP
        for t in range(config.step_count):
            # STEP
            step_counter += 1

            action = select_action(qnet, state, epsilon)

            next_state, reward, done = env.step(action)

            # print(
            #     f"t={t} | state={state} | action={action} | reward={reward} | next_state={next_state}"
            # )

            # LEARN 2
            buffer.store(state, action, reward, next_state, done)

            if len(buffer) >= config.buffer_min_train:
                states, actions, rewards, next_states, dones = buffer.sample(
                    config.batch_size
                )
                q_values = qnet(states)
                q_value = q_values.gather(
                    1, actions.unsqueeze(1)
                )  # for i in len, select q_values[i, actions[i]]
                q_value = q_value.squeeze(1)  # from (len, 1) to (len,)

                with torch.no_grad():
                    best_next_actions = qnet(next_states).argmax(
                        dim=1
                    )  # actions e.g. 0 1 2
                    next_q_values = target_qnet(next_states)
                    # dim=x COLLAPSES that dimension, removing it, THEN apply the operation
                    # OLD: max_next_q_values = next_q_values.max(dim=1)[0]
                    # NEW:
                    max_next_q_values = next_q_values.gather(
                        1, best_next_actions.unsqueeze(1)
                    )  # for i in len, select next_q_values[i, best_next_actions[i]]
                    max_next_q_values = max_next_q_values.squeeze(1)
                    # target = r + γ max_a′ Q(s′, a′)
                    target = (
                        rewards + config.gamma * max_next_q_values * (~dones).float()
                    )
                    # ~dones just makes it so we only add r for the terminal step

                loss = loss_fn(q_value, target)
                loss_array.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target
            if step_counter % config.target_update_freq == 0:
                target_qnet.load_state_dict(qnet.state_dict())

            # update epsilon
            epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

            # LEARN 1
            # q_values = qnet(state_to_tensor(state))

            # q_value = q_values[0, action]

            # with torch.no_grad():
            #     next_q_values = qnet(state_to_tensor(next_state))
            #     target = reward + gamma * next_q_values.max()

            # loss = loss_fn(q_value, target)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # REWARD
            total_reward += reward
            state = next_state

            if done:
                break

        print("Total reward:", total_reward)
        print("Episode finished")
        reward_array.append(total_reward)

    rewards = np.array(reward_array)

    window = 100  # try 25, 50, or 100
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.plot(rewards, alpha=0.3, label="Episode reward")
    plt.plot(
        range(window - 1, len(rewards)),
        moving_avg,
        label=f"{window}-episode moving average",
    )

    plt.axhline(
        y=4.8, linestyle="--", linewidth=2, label="Optimal expected reward (4.8)"
    )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward (Smoothed)")
    plt.legend()
    plt.show()

    print(f"\nExpected reward from optimal policy pi(just go right) = 4.8")

    losses = np.array(loss_array)

    window = 100  # try 25, 50, or 100
    loss_moving_avg = np.convolve(losses, np.ones(window) / window, mode="valid")

    plt.plot(losses, alpha=0.3, label="Training loss")
    plt.plot(
        range(window - 1, len(losses)),
        loss_moving_avg,
        label=f"{window}-step moving average",
    )

    plt.xlabel("Training step")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss (Smoothed)")
    plt.legend()
    plt.show()
