# sac_cheetah/train.py

# observation breakdown (17 dims):")
#   [0]: z-coordinate of torso (height)")
#   [1]: y-rotation angle of torso (pitch)")
#   [2:8]: 6 joint angles")
#   [8]: x-velocity of torso")
#   [9]: z-velocity of torso")
#   [10]: y-angular velocity of torso")
#   [11:17]: 6 joint angular velocities")

# action breakdown
#   six torques
#   Box(-1.0, 1.0, (6,), float32)

# obs: z, p, th_bar | xdot, zdot, p_dot, th_bar_dot
# (x excluded so policy generalises to joints not exact position)
# action: tau_bar
# reward: r = xdot - 0.1 * ||action||^2
# no termination, cheetah can flip/fail silently
# truncates at 1000 steps

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import gymnasium as gym
import os

# replay buffer just like dqn!
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_LAYER_NODES_ACTOR = 256
HIDDEN_LAYER_NODES_CRITIC = 256
ACTION_HIGH = 1.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# DATA CLASSES
class Config:
    total_timesteps: int = 300000
    buffer_size: int = 100000
    min_buffer_train: int = 1000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # soft target update
    lr: float = 3e-4
    # -dim(action) is a common heuristic, you have 6 actions
    target_entropy: float = -6.0


# HELPER FUNCTIONS
def state_to_tensor(state):
    if isinstance(state, np.ndarray):
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    return torch.tensor([state], dtype=torch.float32)


# CLASSES
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.bool)

        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_actor, log_std_min, log_std_max):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_actor),
            nn.ReLU(),
            nn.Linear(hidden_actor, hidden_actor),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_actor, action_dim)
        self.log_std_head = nn.Linear(hidden_actor, action_dim)
        # both mean and std are trained (2 heads)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = self.net(state)

        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std_head(x), self.log_std_min, self.log_std_max)

        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()

        dist = torch.distributions.Normal(mu, std)
        # reparametrization trick with rsample
        z = dist.rsample()  # sample with gradients

        # tanh squash
        squash = torch.tanh(z)
        action = torch.tanh(z) * ACTION_HIGH

        # log prob with tanh correction same as ppo
        log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - squash.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_critic):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_critic),
            nn.ReLU(),
            nn.Linear(hidden_critic, hidden_critic),
            nn.ReLU(),
            nn.Linear(hidden_critic, 1),
        )

    def forward(self, state, action):
        # concentate state and action
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_critic):
        super().__init__()
        self.q1 = Critic(state_dim, action_dim, hidden_critic)
        self.q2 = Critic(state_dim, action_dim, hidden_critic)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)


if __name__ == "__main__":
    env = gym.make("HalfCheetah-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    config = Config()

    actor = Actor(
        state_dim, action_dim, HIDDEN_LAYER_NODES_ACTOR, LOG_STD_MIN, LOG_STD_MAX
    )
    critic = TwinCritic(state_dim, action_dim, HIDDEN_LAYER_NODES_CRITIC)

    # target network like dqn
    target_critic = TwinCritic(state_dim, action_dim, HIDDEN_LAYER_NODES_CRITIC)
    target_critic.load_state_dict(critic.state_dict())

    # learnable log_alpha (log so alpha stays positive)
    log_alpha = torch.zeros(1, requires_grad=True)

    actor_optimizer = optim.Adam(actor.parameters(), lr=config.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.lr)
    alpha_optimizer = optim.Adam([log_alpha], lr=config.lr)

    # replay buffer like dqn
    buffer = ReplayBuffer(config.buffer_size)

    # logging
    total_reward_array = []
    obs, info = env.reset()
    ep_return = 0.0

    # timestep loop
    for step in trange(config.total_timesteps):
        # before the buffer is filled we do random
        if step < config.min_buffer_train:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _ = actor.sample(state_to_tensor(obs))
                # we do not need log prob here
                action = action.squeeze(0).numpy()
                # squeeze to vector and make a vector so we can pass to gym

        # step env
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # store transition
        buffer.store(obs, action, reward, next_obs, done)

        ep_return += reward
        obs = next_obs

        if done:
            total_reward_array.append(ep_return)
            ep_return = 0.0
            obs, info = env.reset()

        if step >= config.min_buffer_train:
            states, actions, rewards, next_states, dones = buffer.sample(
                config.batch_size
            )
            # alpha for training
            alpha = log_alpha.exp().detach()
            if step % 10000 == 0:
                print(f"Step {step}, alpha: {log_alpha.exp().item():.4f}")

            # CRITIC UPDATE
            with torch.no_grad():
                next_actions, next_log_probs = actor.sample(next_states)

                q1_target, q2_target = target_critic(next_states, next_actions)
                q_target = torch.min(q1_target, q2_target)

                # SAC target: r + γ * (Q - α * log_prob)
                # The entropy term encourages exploration
                target = (
                    rewards
                    + config.gamma
                    * (q_target - alpha * next_log_probs)
                    * (~dones).float()
                )

            # current q value
            q1, q2 = critic(states, actions)

            # loss = mse mean sum
            critic_loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # ACTOR UPDATE
            # sample fresh actions for gradients and training?
            actions_, log_probs = actor.sample(states)

            q1_, q2_ = critic(states, actions_)
            q_ = torch.min(q1_, q2_)

            # Actor wants to maximize Q and maximize entropy (minimize log_prob)
            # Loss = α * log_prob - Q  (minimize this = maximize Q - α * log_prob)
            actor_loss = (alpha * log_probs - q_).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # ALPHA UPDATE
            # L = -a(logp + logaT)
            alpha_loss = -(
                log_alpha.exp() * (log_probs.detach() + config.target_entropy)
            ).mean()

            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            # SOFT TARGET UPDATE EACH STEP
            with torch.no_grad():
                for param, target_param in zip(
                    critic.parameters(), target_critic.parameters()
                ):
                    target_param.data.copy_(
                        (1 - config.tau) * target_param.data + config.tau * param.data
                    )

    # PLOT REWARD
    total_rewards = np.array(total_reward_array)
    window = 20
    if len(total_rewards) >= window:
        moving_avg = np.convolve(total_rewards, np.ones(window) / window, mode="valid")
        plt.plot(total_rewards, alpha=0.3, label="Training")
        plt.plot(
            range(window - 1, len(total_rewards)),
            moving_avg,
            label=f"{window}-ep moving avg",
        )
    else:
        plt.plot(total_rewards, label="Training")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SAC Cheetah")
    plt.legend()
    plt.show()

    # SAVE
    os.makedirs("models_cheetah", exist_ok=True)
    torch.save(actor.state_dict(), "models_cheetah/cheetah_sac_actor.pth")
    torch.save(critic.state_dict(), "models_cheetah/cheetah_sac_critic.pth")
    print("Models saved")
