# ppo_vec_cheetah/train.py

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import gymnasium as gym
import os

import torch
import torch.nn as nn
import torch.optim as optim


# DATA CLASSES
class Config:
    seed: int = 19
    max_grad_norm: float = 0.5
    total_timesteps: int = 200_000
    rollout_steps: int = 2_048
    minibatch_size: int = 256
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon: float = 0.2
    lr: float = 3e-4
    hidden_actor: int = 256
    hidden_critic: int = 256
    log_std_min: float = -2.0
    log_std_max: float = 1.0
    action_high: float = 1.0
    ent_coef: float = 0.01
    vf_coef: float = 0.5


# HELPER FUNCTIONS
def state_to_tensor(state, device):
    if isinstance(state, np.ndarray):
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    return torch.tensor([state], dtype=torch.float32, device=device)


# CLASSES
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, config: Config):
        super().__init__()
        self.config = config

        self.actor_body = nn.Sequential(
            nn.Linear(state_dim, self.config.hidden_actor),
            nn.Tanh(),
            nn.Linear(self.config.hidden_actor, self.config.hidden_actor),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(self.config.hidden_actor, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, self.config.hidden_critic),
            nn.Tanh(),
            nn.Linear(self.config.hidden_critic, self.config.hidden_critic),
            nn.Tanh(),
            nn.Linear(self.config.hidden_critic, 1),
        )

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def _get_dist(self, obs):
        h = self.actor_body(obs)
        mu = self.mu_head(h)
        log_std = self.log_std.clamp(self.config.log_std_min, self.config.log_std_max)
        std = log_std.exp()
        return torch.distributions.Normal(mu, std)

    def get_action_and_logprob(self, obs):
        dist = self._get_dist(obs)

        z = dist.rsample()
        a = torch.tanh(z)
        action_env = a * self.config.action_high

        logprob = dist.log_prob(z).sum(dim=-1)
        logprob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)

        # use action and store log prob old in rollout
        return action_env, logprob

    def logprob_of_action(self, obs, action_env):
        dist = self._get_dist(obs)

        a = torch.clamp(action_env / self.config.action_high, -0.999, 0.999)
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))

        logprob = dist.log_prob(z).sum(dim=-1)
        logprob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        # use log prob new and use entropy in training (also use logprob old)
        return logprob, entropy


if __name__ == "__main__":
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed random
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = gym.make("HalfCheetah-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor_critic = ActorCritic(state_dim, action_dim, config).to(device=device)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=config.lr)

    # pre-allocated tensors
    obs_buff = torch.zeros((config.rollout_steps, state_dim), device=device)
    action_buff = torch.zeros((config.rollout_steps, action_dim), device=device)
    reward_buff = torch.zeros(config.rollout_steps, device=device)
    done_buff = torch.zeros(config.rollout_steps, device=device)
    logprob_buff = torch.zeros(config.rollout_steps, device=device)
    value_buff = torch.zeros(config.rollout_steps, device=device)

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
                obs_tensor = state_to_tensor(obs, device=device)
                action_env, logprob = actor_critic.get_action_and_logprob(obs_tensor)
                value = actor_critic.value(obs_tensor)
                action_np = action_env.squeeze().cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # store in buffers
            obs_buff[step] = obs_tensor.squeeze(0)
            action_buff[step] = action_env.squeeze(0)
            reward_buff[step] = reward
            done_buff[step] = 1.0 if done else 0.0
            logprob_buff[step] = logprob.squeeze(0)
            value_buff[step] = value.squeeze(0)

            ep_return += reward
            obs = next_obs

            if done:
                total_reward_array.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()

        # GAE COMPUTATION (with proper done handling)
        with torch.no_grad():
            next_value = actor_critic.value(
                state_to_tensor(obs, device=device)
            ).squeeze()

        advantages = torch.zeros(config.rollout_steps, device=device)
        gae = 0.0

        for t in reversed(range(config.rollout_steps)):
            if t == config.rollout_steps - 1:
                next_nonterminal = 1.0 - done_buff[t]
                next_val = next_value
            else:
                next_nonterminal = 1.0 - done_buff[t + 1]
                next_val = value_buff[t + 1]

            delta = (
                reward_buff[t]
                + config.gamma * next_val * next_nonterminal
                - value_buff[t]
            )
            gae = delta + config.gamma * config.gae_lambda * next_nonterminal * gae
            advantages[t] = gae

        returns = advantages + value_buff

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # MINIBATCH TRAINING (batched, not per-sample)
        for epoch in range(config.update_epochs):
            indices = torch.randperm(config.rollout_steps)

            for start in range(0, config.rollout_steps, config.minibatch_size):
                mb_indices = indices[start : start + config.minibatch_size]

                # gather minibatch (all at once)
                mb_obs = obs_buff[mb_indices]
                mb_act = action_buff[mb_indices]
                mb_logprob_old = logprob_buff[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # forward pass on entire minibatch
                logprob_new, entropy = actor_critic.logprob_of_action(mb_obs, mb_act)
                values_new = actor_critic.value(mb_obs)

                # policy loss
                ratio = torch.exp(logprob_new - mb_logprob_old)
                surrogate = ratio * mb_advantages
                clipped_surrogate = (
                    torch.clamp(ratio, 1 - config.epsilon, 1 + config.epsilon)
                    * mb_advantages
                )
                policy_loss = -torch.min(surrogate, clipped_surrogate).mean()

                # value loss
                value_loss = ((mb_returns - values_new) ** 2).mean()

                # entropy loss
                entropy_loss = -entropy.mean()

                # total loss and update
                total_loss = (
                    policy_loss
                    + config.vf_coef * value_loss
                    + config.ent_coef * entropy_loss
                )

                ac_optimizer.zero_grad()
                total_loss.backward()
                # clip gradients
                nn.utils.clip_grad_norm_(
                    actor_critic.parameters(), config.max_grad_norm
                )
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
