# sac_cheetah/print_vectorised.py

import gymnasium as gym
import numpy as np

NUM_ENVS = 4


def make_env():
    return gym.make("HalfCheetah-v5")


# create VECTORISED environment
env = gym.vector.SyncVectorEnv([make_env for _ in range(NUM_ENVS)])

print("Vectorised HalfCheetah (SyncVectorEnv)")
print("https://gymnasium.farama.org/api/vector/")

# reset
obs, info = env.reset()

print("\nobs shape:", obs.shape)  # (num_envs, obs_dim)
print("obs dtype:", obs.dtype)
print("obs[0]:", obs[0])

# spaces
print("\nsingle action space:", env.single_action_space)
print("single observation space:", env.single_observation_space)

# sample one action PER ENV
actions = np.stack([env.single_action_space.sample() for _ in range(NUM_ENVS)])

print("\nactions shape:", actions.shape)

# step
next_obs, reward, terminated, truncated, info = env.step(actions)

print("\nnext_obs shape:", next_obs.shape)
print("reward shape:", reward.shape)
print("terminated shape:", terminated.shape)
print("truncated shape:", truncated.shape)

print("\nreward:", reward)
print("terminated:", terminated)
print("truncated:", truncated)

# short rollout
total_rewards = np.zeros(NUM_ENVS)

for _ in range(10):
    actions = np.stack([env.single_action_space.sample() for _ in range(NUM_ENVS)])
    obs, reward, terminated, truncated, info = env.step(actions)
    total_rewards += reward

print("\ntotal rewards per env:", total_rewards)

env.close()
