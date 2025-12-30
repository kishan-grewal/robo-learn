# ppo_vec_cheetah/train_vectorised.py

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import gymnasium as gym
import os

import torch
import torch.nn as nn
import torch.optim as optim


class Config:
    # higher level parameters
    seed: int = 19
    action_high: float = 1.0
    total_timesteps: int = 1_000_000
    # rollout_steps: int = 2_048  # 200_000 steps but we stop to train every 2_048
    # rollout_steps = num_envs * rollouts_per_env
    num_envs: int = 32  # number of parallel environments
    rollouts_per_env: int = 128  # number of rollouts per parallel env
    minibatch_size: int = 256  # we train on those 2_048 in 256 chunks

    # training
    update_epochs: int = 10  # we train multiple times on each minibatch
    lr: float = 3e-4  # universal learning rate
    max_grad_norm: float = 0.5  # numerical stability

    # objective
    gamma: float = 0.99

    # gae and surrogate
    gae_lambda: float = 0.95  # discount surprise (gae)
    epsilon: float = 0.2  # clip ratio for policy change

    # actorcritic network
    hidden_actor: int = 256
    hidden_critic: int = 256
    log_std_min: float = -2.0
    log_std_max: float = 1.0

    # total loss coefficients
    ent_coef: float = 0.01
    vf_coef: float = 0.5


# DONT NEED anymore, we always have batches due to env dimension
# def state_to_tensor(state, device):
#     if isinstance(state, np.ndarray):
#         return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#     return torch.tensor([state], dtype=torch.float32, device=device)


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
        """
        TODO #1: Sample action and compute log probability with tanh squashing

        Steps:
        1. Get the Normal distribution from _get_dist
        2. Sample z using rsample() (reparameterized for gradients)
        3. Apply tanh to get a in [-1, 1]
        4. Scale to action_high for environment
        5. Compute log_prob of z from the Normal distribution (sum over action dims)
        6. Apply tanh correction: subtract log(1 - tanh(z)^2) for each action dim
           (This is the Jacobian correction for change of variables)
        7. Compute entropy from the distribution

        Return: action_env, logprob, entropy, z

        Hint: The tanh correction formula is:
            logprob = dist.log_prob(z).sum(-1) - log(1 - a^2 + eps).sum(-1)

        Why? When you transform a random variable through tanh, the probability
        density changes. This correction accounts for that change.
        """
        dist = self._get_dist(obs)

        # reparametrised
        z = dist.rsample()

        a = torch.tanh(z)

        action_env = a * self.config.action_high

        # literally log of policy (probability of action with given state)
        log_prob = dist.log_prob(z).sum(dim=-1)

        # adjust for tanh transformation of probabilities
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)

        return action_env, log_prob

    def logprob_of_action(self, obs, action_env):
        """
        TODO #2: Compute log probability of a previously taken action

        Steps:
        1. Get the Normal distribution from _get_dist
        2. Recover 'a' from action_env (divide by action_high, clamp to avoid atanh explosion)
        3. Recover 'z' from 'a' using inverse tanh: z = atanh(a) = 0.5 * (log(1+a) - log(1-a))
        4. Compute log_prob of z, apply same tanh correction as above
        5. Compute entropy

        Return: logprob, entropy

        Why do we need this? During training, we recompute log probs for actions
        stored in the buffer, using the CURRENT policy (which has been updated).
        This lets us compute the probability ratio for PPO clipping.
        """
        dist = self._get_dist(obs)

        # now we literally inverse the above function z -> action, logprob and we go from action -> logprob

        # action_env = a * action_high
        a = action_env / self.config.action_high

        # numerical instability
        a = torch.clamp(a, -0.999, 0.999)

        # a = tanh(z)
        # z = arctanh(a)
        # z = 1/2 * ln(1+x / 1-x)
        z = 0.5 * (torch.log1p(+a) - torch.log1p(-a))
        # log1p(x) is literally a more accurate log(1+x) for small x

        # literally log of policy (probability of action with given state)
        log_prob = dist.log_prob(z).sum(dim=-1)

        # adjust for tanh transformation of probabilities
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """
    TODO #3: Compute Generalized Advantage Estimation

    Args:
        rewards: tensor of shape (rollout_steps,)
        values: tensor of shape (rollout_steps,) - V(s_t) for each step
        dones: tensor of shape (rollout_steps,) - 1.0 if episode ended, 0.0 otherwise
        next_value: scalar tensor - V(s_T) for the state after the last step
        gamma: discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: tensor of shape (rollout_steps,)

    The GAE formula (work backwards from t=T-1 to t=0):
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

    Key insight: GAE is a weighted average of n-step advantage estimates.
    lambda=0 gives 1-step TD, lambda=1 gives full Monte Carlo.
    lambda=0.95 is a good balance of bias and variance.

    Why (1 - done)? If the episode ended, there's no future value to bootstrap from.
    The next state's value should not contribute to the current advantage.
    """
    # rewards: (rollouts_per_env, num_envs)
    # values: (rollouts_per_env, num_envs)
    # dones: (rollouts_per_env, num_envs)
    # next_value: (num_envs,)

    rollouts_per_env = rewards.shape[0]  # dim 0 and 1
    num_envs = rewards.shape[1]

    advantages = torch.zeros_like(rewards)

    gae = torch.zeros(num_envs, device=rewards.device)  # isolated gae per environment

    for t in reversed(range(rollouts_per_env)):
        if t == rollouts_per_env - 1:
            next_v = next_value
        else:
            next_v = values[t + 1]

        # these all broadcast correctly now, same formula but vectorised
        # surprise per env
        delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
        # weighted surprise per env
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        # store, go recursively
        advantages[t] = gae

    return advantages


def compute_policy_loss(logprob_new, logprob_old, advantages, epsilon):
    """
    TODO #4: Compute PPO clipped policy loss

    Args:
        logprob_new: log probabilities under current policy
        logprob_old: log probabilities under old policy (from buffer)
        advantages: advantage estimates
        epsilon: clipping parameter (e.g., 0.2)

    Returns:
        policy_loss: scalar tensor (to be minimized)

    The PPO-clip objective:
        ratio = exp(logprob_new - logprob_old)  # = pi_new(a|s) / pi_old(a|s)
        surrogate = ratio * advantage
        clipped_surrogate = clip(ratio, 1-eps, 1+eps) * advantage
        loss = -min(surrogate, clipped_surrogate)  # negative because we minimize

    Why clipping? Prevents the policy from changing too much in one update.
    If ratio > 1+eps, the clipped version limits the "credit" for that action.
    If ratio < 1-eps, same thing in the other direction.

    Why min? We're pessimistic - take the worse of the two estimates.
    This prevents the policy from exploiting the advantage estimate.
    """

    # ratio = p'(a) / p(a) where we took the action a with p and recalculate p'(a) as if we took action a with p'
    # ratio = pi_new(a) / pi_old(a) where pi_old is what we decided on a with
    # ratio = exp(log_pi_new - log_pi_old)
    ratio = torch.exp(logprob_new - logprob_old)
    surrogate = ratio * advantages

    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    clipped_surrogate = clipped_ratio * advantages  # dont clip advantages ever

    # maximise min(surrogate, clipped_surrogate)
    # minimise -1 * min(surrogate, clipped_surrogate)
    policy_loss = -torch.min(surrogate, clipped_surrogate).mean()
    return policy_loss


def compute_entropy_loss(entropy):
    """
    TODO #5: Compute entropy loss

    Args:
        entropy: tensor of entropy values for each sample

    Returns:
        entropy_loss: scalar tensor

    Question to answer in your implementation:
    - Why is entropy loss NEGATIVE (we return -entropy.mean())?
    - What happens to exploration if we add positive entropy to the objective?

    Hint: We MINIMIZE total_loss. We WANT higher entropy (more exploration).
    So entropy should be SUBTRACTED from the loss, or equivalently,
    we return negative entropy as "entropy_loss" and ADD it to total_loss.
    """
    return -entropy.mean()


if __name__ == "__main__":
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    def make_env():
        return gym.make("HalfCheetah-v5")

    # sync vector env for 16 parallel environments
    env = gym.vector.SyncVectorEnv([make_env for _ in range(config.num_envs)])
    # we place [t, env, ..] not [env, t, ..] because
    # then we can index by time easier, which we index by more than env

    state_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]

    actor_critic = ActorCritic(state_dim, action_dim, config).to(device=device)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=config.lr)

    # predefine s, a, r, done, log_pi, v tensors for speed
    obs_buff = torch.zeros(
        (config.rollouts_per_env, config.num_envs, state_dim), device=device
    )
    action_buff = torch.zeros(
        (config.rollouts_per_env, config.num_envs, action_dim), device=device
    )
    reward_buff = torch.zeros(config.rollouts_per_env, config.num_envs, device=device)
    done_buff = torch.zeros(config.rollouts_per_env, config.num_envs, device=device)
    logprob_buff = torch.zeros(config.rollouts_per_env, config.num_envs, device=device)
    value_buff = torch.zeros(config.rollouts_per_env, config.num_envs, device=device)

    total_reward_array = []
    global_step = 0
    rollout_steps = config.num_envs * config.rollouts_per_env
    num_updates = config.total_timesteps // rollout_steps

    obs, _ = env.reset()
    episode_rewards = np.zeros(config.num_envs)

    for update in trange(num_updates):
        # ROLLOUT COLLECTION (this part is given - it's just stepping the envs)
        for step in range(config.rollouts_per_env):
            global_step += 1

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                action_env, logprob = actor_critic.get_action_and_logprob(obs_tensor)
                value = actor_critic.value(obs_tensor)
                action_np = action_env.squeeze().cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            # new done handling for many envs
            done = np.logical_or(terminated, truncated)

            obs_buff[step] = obs_tensor
            action_buff[step] = action_env
            reward_buff[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            done_buff[step] = torch.tensor(done, dtype=torch.float32, device=device)
            logprob_buff[step] = logprob
            value_buff[step] = value

            obs = next_obs
            episode_rewards += reward

            # check which envs terminated or truncated
            for i in range(config.num_envs):
                if done[i]:
                    total_reward_array.append(episode_rewards[i])
                    episode_rewards[i] = 0.0

        # GAE COMPUTATION
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            next_value = actor_critic.value(obs_tensor)

        advantages = compute_gae(
            reward_buff,
            value_buff,
            done_buff,
            next_value,
            config.gamma,
            config.gae_lambda,
        )
        returns = advantages + value_buff
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # flatten here before taking random minibatches
        obs_flat = obs_buff.reshape(-1, state_dim)
        action_flat = action_buff.reshape(-1, action_dim)
        logprob_flat = logprob_buff.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)

        # calculate rollout_steps for mini batch training
        rollout_steps = config.num_envs * config.rollouts_per_env

        # MINIBATCH TRAINING: NOW FLATTENED DUE TO MANY PARALLEL ENVS
        for epoch in range(config.update_epochs):
            indices = torch.randperm(rollout_steps)

            for start in range(0, rollout_steps, config.minibatch_size):
                indices_mb = indices[start : start + config.minibatch_size]

                obs_mb = obs_flat[indices_mb]
                action_mb = action_flat[indices_mb]
                logprob_old_mb = logprob_flat[indices_mb]
                advantages_mb = advantages_flat[indices_mb]
                returns_mb = returns_flat[indices_mb]

                logprob_new, entropy = actor_critic.logprob_of_action(obs_mb, action_mb)
                values_new = actor_critic.value(obs_mb)

                policy_loss = compute_policy_loss(
                    logprob_new, logprob_old_mb, advantages_mb, config.epsilon
                )
                value_loss = ((returns_mb - values_new) ** 2).mean()
                entropy_loss = compute_entropy_loss(entropy)

                total_loss = (
                    policy_loss  # [1 v e just scalar knobs for loss in training]
                    + config.vf_coef * value_loss
                    + config.ent_coef * entropy_loss
                )
                # maximise policy_objective + value_objective + entropy
                # = minimise policy_loss + value_loss + entropy_loss

                ac_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    actor_critic.parameters(), config.max_grad_norm
                )
                ac_optimizer.step()

    # PLOTTING
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
    plt.title("Vectorised PPO on HalfCheetah-v5")
    plt.legend()
    plt.show()

    os.makedirs("models_cheetah", exist_ok=True)
    torch.save(actor_critic.state_dict(), "models_cheetah/cheetah_ppo_vec_ac.pth")
    print("Model saved to models_cheetah/cheetah_ppo_vec_ac.pth")
