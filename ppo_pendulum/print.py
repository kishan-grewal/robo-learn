# ppo_pendulum/print.py

import gymnasium as gym
import numpy as np

# create gym environment
env = gym.make(
    "Pendulum-v1"
)  # https://gymnasium.farama.org/environments/classic_control/pendulum/
print("https://gymnasium.farama.org/environments/classic_control/pendulum/")

# reset and inspect
obs, info = env.reset()

print(
    """
# notes
# obs gives cos and sin of theta + thetadot, not raw theta
# this is because there is a discontinuity in theta at the bottom
# radius = 1, theta = [-pi, pi] where 0 is at the top, the target and +-pi is at the bottom
# there are no positive rewards, just a negative reward each step
# R = -(th^2 + 0.1 * thdot^2 + 0.001 * tau^2)
# max torque = +- 2 Nm and max thdot = +- 8 rad/s
      """
)

print("obs:", obs)  # [cos(theta), sin(theta), theta_dot]
print("obs shape:", obs.shape)  # (3,)
print("obs dtype:", obs.dtype)  # float32

print("\naction space:", env.action_space)  # Box(-2.0, 2.0, (1,), float32)
print("action shape:", env.action_space.shape)  # (1,) - continuous!
print("action low:", env.action_space.low)  # [-2.]
print("action high:", env.action_space.high)  # [2.]
# No env.action_space.n - continuous actions don't have discrete count

print("\nobservation space:", env.observation_space)
print("observation dimensions:", env.observation_space.shape[0])  # 3 dim
print("obs low:", env.observation_space.low)  # [-1, -1, -8]
print("obs high:", env.observation_space.high)  # [1, 1, 8]

action = env.action_space.sample()  # sample continuous action
next_obs, reward, terminated, truncated, info = env.step(action)

print("\naction taken:", action)  # e.g. [0.734] - continuous torque
print("next observation:", next_obs)
print("reward:", reward)  # Negative, based on angle and effort
print("terminated:", terminated)  # Always False for Pendulum
print("truncated (time limit):", truncated)
print("info:", info)

env.reset()
total_reward = 0
done = False
steps = 0

while not done and steps < 200:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    steps += 1

print("\nsteps taken:", steps)
print("total reward:", total_reward)
# Reward is negative: -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
# Best possible reward per step is 0 (upright, no velocity, no torque)
# Typical random policy gets around -1000 to -1500 per episode

env.close()

print(
    """\nKey differences from CartPole:
\n1. CONTINUOUS action space: torque in [-2, 2], not discrete left/right
\n2. Never terminates early - always runs for 200 steps (truncation only)
\n3. Reward is NEGATIVE: penalizes angle from upright, angular velocity, and torque
\n4. Goal: swing up and balance the pendulum upright (theta=0)
\n5. Observation is [cos(theta), sin(theta), theta_dot] not raw angle
\n6. Max episode reward ~0 (perfect), random policy gets ~-1000 to -1500\n"""
)

# notes
# obs gives cos and sin of theta + thetadot, not raw theta
# this is because there is a discontinuity in theta at the bottom
# radius = 1, theta = [-pi, pi] where 0 is at the top, the target and +-pi is at the bottom
# there are no positive rewards, just a negative reward each step
# R = -(th^2 + 0.1 * thdot^2 + 0.001 * tau^2)
# max torque = +- 2 Nm and max thdot = +- 8 rad/s
