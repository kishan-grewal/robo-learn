# sac_pendulum/train.py

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import gymnasium as gym
import os

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_LAYER_NODES_CRITIC = 256


# DATA CLASSES
class Config:
    pass


# CLASSES
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_LAYER_NODES_CRITIC),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_NODES_CRITIC, HIDDEN_LAYER_NODES_CRITIC),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_NODES_CRITIC, 1),
        )

    def forward(self, state, action):
        # concentate state and action
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


if __name__ == "__main__":
    critic = Critic(state_dim=3, action_dim=1)
    state = torch.randn(1, 3)  # batch of 1, Pendulum has 3 obs dims
    action = torch.randn(1, 1)  # batch of 1, Pendulum has 1 action dim
    q_value = critic(state, action)
    print(q_value.shape)  # (1, 1)
