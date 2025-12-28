# sac_cheetah/print.py

import gymnasium as gym
import numpy as np

# create gym environment
env = gym.make(
    "HalfCheetah-v5"
)  # https://gymnasium.farama.org/environments/mujoco/half_cheetah/
print("https://gymnasium.farama.org/environments/mujoco/half_cheetah/")

# reset and inspect
obs, info = env.reset()

print(
    """
# notes
# HalfCheetah is a 2D robot with 9 rigid links (including torso) and 6 actuated joints
# goal: run forward (positive x direction) as fast as possible
# reward = forward_velocity - 0.1 * control_cost
# forward_velocity: how fast the cheetah moves in x direction
# control_cost: penalizes large actions (sum of squared actions)
# no termination condition - runs for 1000 steps by default
      """
)

print("obs:", obs)
print("obs shape:", obs.shape)  # (17,)
print("obs dtype:", obs.dtype)  # float32

print("\naction space:", env.action_space)  # Box(-1.0, 1.0, (6,), float32)
print("action shape:", env.action_space.shape)  # (6,) - 6 joint torques
print("action low:", env.action_space.low)  # [-1, -1, -1, -1, -1, -1]
print("action high:", env.action_space.high)  # [1, 1, 1, 1, 1, 1]

print("\nobservation space:", env.observation_space)
print("observation dimensions:", env.observation_space.shape[0])  # 17 dim

print("\nobservation breakdown (17 dims):")
print("  [0]: z-coordinate of torso (height)")
print("  [1]: y-rotation angle of torso (pitch)")
print("  [2:8]: 6 joint angles")
print("  [8]: x-velocity of torso")
print("  [9]: z-velocity of torso")
print("  [10]: y-angular velocity of torso")
print("  [11:17]: 6 joint angular velocities")

action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)

print("\naction taken:", action)
print("next observation shape:", next_obs.shape)
print("reward:", reward)
print("terminated:", terminated)
print("truncated (time limit):", truncated)
print("info:", info)

env.reset()
total_reward = 0
done = False
steps = 0

while not done and steps < 1000:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    steps += 1

print("\nsteps taken:", steps)
print("total reward:", total_reward)
# Random policy gets around -200 to +200 (mostly from random forward/backward movement)
# Good policy gets 3000-5000+
# Expert policy can get 10000+

env.close()

print(
    """\nKey differences from Pendulum:
\n1. Much higher dimensional: 17D state vs 3D, 6D action vs 1D
\n2. POSITIVE rewards possible: get rewarded for running forward
\n3. Action range is [-1, 1] not [-2, 2]
\n4. Episode length is 1000 steps not 200
\n5. Multiple joints to coordinate (6 actuators)
\n6. Forward velocity directly rewarded, not angle-based
\n7. Random policy ~0 reward, good policy 3000-5000+, expert 10000+\n"""
)

# observation details:
# - positions and velocities of all body parts
# - x-position is NOT included (so policy can't memorize positions)
# - reward is velocity-based so policy learns to run, not reach a position
