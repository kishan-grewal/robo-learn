# ppo_pendulum/ppo_pendulum_gpu.py
# PPO + GAE for Gymnasium Pendulum-v1 (GPU-friendly, stable)
# - tanh-squashed Gaussian policy with correct log-prob correction
# - GAE(λ) advantages
# - minibatch PPO updates over a rollout of N steps
# - proper CPU<->GPU boundary: only the env action goes to CPU

import os
import math
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    seed: int = 1
    total_timesteps: int = 400_000
    rollout_steps: int = 2048          # collect this many steps per update
    update_epochs: int = 10            # PPO epochs per rollout
    minibatch_size: int = 256          # must divide rollout_steps
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.0             # start with 0; add 0.001 if you want more exploration
    max_grad_norm: float = 0.5
    log_std_min: float = -2.0
    log_std_max: float = 1.0


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


# ----------------------------
# Networks
# ----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, action_high: float, cfg: Config):
        super().__init__()
        self.action_high = float(action_high)
        self.cfg = cfg

        hidden = 64
        self.actor_body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def _clamped_std(self) -> torch.Tensor:
        log_std = self.log_std.clamp(self.cfg.log_std_min, self.cfg.log_std_max)
        return log_std.exp()

    def get_action_and_logprob(self, obs: torch.Tensor):
        """
        Returns:
          action_env: squashed and scaled action in [-action_high, action_high]
          logprob: log pi(a|s) with tanh correction
          entropy: entropy of the *pre-squash* Normal (useful as a regulariser; not exact env-entropy)
          z: pre-tanh action (for debugging)
        """
        h = self.actor_body(obs)
        mu = self.mu_head(h)
        std = self._clamped_std()
        dist = torch.distributions.Normal(mu, std)

        z = dist.rsample()  # (B, act_dim)
        a = torch.tanh(z)
        action_env = a * self.action_high

        # log prob with tanh correction:
        # log pi(a) = log N(z; mu, std) - sum log(1 - tanh(z)^2)
        logprob = dist.log_prob(z).sum(dim=-1)
        logprob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return action_env, logprob, entropy, z

    def logprob_of_action(self, obs: torch.Tensor, action_env: torch.Tensor):
        """
        Compute log pi(a|s) for already-squashed-and-scaled env action.
        We invert the squash: a = tanh(z)*high -> tanh(z)=a/high -> z = atanh(a/high)
        """
        h = self.actor_body(obs)
        mu = self.mu_head(h)
        std = self._clamped_std()
        dist = torch.distributions.Normal(mu, std)

        a = torch.clamp(action_env / self.action_high, -0.999, 0.999)
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh(a)

        logprob = dist.log_prob(z).sum(dim=-1)
        logprob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logprob, entropy


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    env = gym.make("Pendulum-v1")
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
    env.observation_space.seed(cfg.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_high = float(env.action_space.high[0])

    ac = ActorCritic(obs_dim, act_dim, action_high, cfg).to(device)
    opt = optim.Adam(ac.parameters(), lr=cfg.lr)

    assert cfg.rollout_steps % cfg.minibatch_size == 0, "minibatch_size must divide rollout_steps"

    # Rollout storage (on GPU)
    obs_buf = torch.zeros((cfg.rollout_steps, obs_dim), device=device)
    act_buf = torch.zeros((cfg.rollout_steps, act_dim), device=device)
    logp_buf = torch.zeros((cfg.rollout_steps,), device=device)
    rew_buf = torch.zeros((cfg.rollout_steps,), device=device)
    done_buf = torch.zeros((cfg.rollout_steps,), device=device)
    val_buf = torch.zeros((cfg.rollout_steps,), device=device)

    # Logging
    ep_returns = []
    recent_return = 0.0
    recent_len = 0

    obs, _ = env.reset()
    global_step = 0
    num_updates = cfg.total_timesteps // cfg.rollout_steps

    for update in trange(num_updates, desc="Updates"):
        # ----------------------------
        # Collect rollout
        # ----------------------------
        for t in range(cfg.rollout_steps):
            global_step += 1
            obs_t = to_tensor(obs, device).view(1, -1)

            with torch.no_grad():
                action_env, logp, _, _ = ac.get_action_and_logprob(obs_t)
                value = ac.value(obs_t)

            # store (GPU)
            obs_buf[t] = obs_t.squeeze(0)
            act_buf[t] = action_env.squeeze(0)
            logp_buf[t] = logp.squeeze(0)
            val_buf[t] = value.squeeze(0)

            # step env (CPU)
            action_np = action_env.squeeze(0).detach().cpu().numpy().astype(np.float32)
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            rew_buf[t] = float(reward)
            done_buf[t] = 1.0 if done else 0.0

            recent_return += float(reward)
            recent_len += 1

            obs = next_obs

            if done:
                ep_returns.append(recent_return)
                recent_return = 0.0
                recent_len = 0
                obs, _ = env.reset()

        # bootstrap value for GAE
        with torch.no_grad():
            next_obs_t = to_tensor(obs, device).view(1, -1)
            next_value = ac.value(next_obs_t).squeeze(0)

        # ----------------------------
        # Compute GAE advantages + returns (on GPU)
        # ----------------------------
        adv_buf = torch.zeros((cfg.rollout_steps,), device=device)
        last_gae = 0.0
        for t in reversed(range(cfg.rollout_steps)):
            if t == cfg.rollout_steps - 1:
                next_nonterminal = 1.0 - done_buf[t]
                next_vals = next_value
            else:
                next_nonterminal = 1.0 - done_buf[t + 1]
                next_vals = val_buf[t + 1]

            delta = rew_buf[t] + cfg.gamma * next_vals * next_nonterminal - val_buf[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            adv_buf[t] = last_gae

        ret_buf = adv_buf + val_buf

        # normalise advantages
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        # ----------------------------
        # PPO update
        # ----------------------------
        batch_inds = torch.arange(cfg.rollout_steps, device=device)
        for _ in range(cfg.update_epochs):
            # shuffle indices on GPU
            perm = batch_inds[torch.randperm(cfg.rollout_steps, device=device)]
            for start in range(0, cfg.rollout_steps, cfg.minibatch_size):
                mb_inds = perm[start : start + cfg.minibatch_size]

                mb_obs = obs_buf[mb_inds]
                mb_act = act_buf[mb_inds]
                mb_old_logp = logp_buf[mb_inds]
                mb_adv = adv_buf[mb_inds]
                mb_ret = ret_buf[mb_inds]

                new_logp, entropy = ac.logprob_of_action(mb_obs, mb_act)
                ratio = torch.exp(new_logp - mb_old_logp)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                values = ac.value(mb_obs)
                value_loss = ((mb_ret - values) ** 2).mean()

                ent_loss = -entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * ent_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                opt.step()

    # ----------------------------
    # Plot + save
    # ----------------------------
    os.makedirs("models", exist_ok=True)
    torch.save(ac.state_dict(), "models/pendulum_ppo_ac.pth")
    print("Saved: models/pendulum_ppo_ac.pth")

    if len(ep_returns) > 0:
        returns = np.array(ep_returns, dtype=np.float32)
        window = 20
        if len(returns) >= window:
            ma = np.convolve(returns, np.ones(window) / window, mode="valid")
            plt.plot(returns, alpha=0.3, label="Episode return")
            plt.plot(np.arange(window - 1, window - 1 + len(ma)), ma, label=f"{window}-ep moving avg")
        else:
            plt.plot(returns, label="Episode return")
        plt.xlabel("Episode")
        plt.ylabel("Return (less negative is better)")
        plt.title("Pendulum-v1 PPO+GAE")
        plt.legend()
        plt.show()
    else:
        print("No episodes finished during training (unexpected).")


if __name__ == "__main__":
    main()
