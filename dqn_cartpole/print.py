import gymnasium as gym
import numpy as np

# create gym environment, now we have some scaffolding
env = gym.make(
    "CartPole-v1"
)  # https://gymnasium.farama.org/environments/classic_control/cart_pole/
print("https://gymnasium.farama.org/environments/classic_control/cart_pole/")

# reset and inspect
obs, info = env.reset()

print("obs:", obs)  # x xdot theta thetadot
print("obs shape:", obs.shape)  # (4,)
print("obs dtype:", obs.dtype)  # float32

print("\naction:", env.action_space)  # 2 actions left or right
print("no. of actions", env.action_space.n)  # 0 left 1 right

print(
    "\nobservation space:", env.observation_space
)  # +- [4.8 m, inf m/s, 24 deg, inf rad/s] are the bounds
print("observation dimensions:", env.observation_space.shape[0])  # 4 dim

action = env.action_space.sample()  # sample -> step in the loop
next_obs, reward, terminated, truncated, info = env.step(action)

print("\naction taken:", action)
print("next observation:", next_obs)
print("reward:", reward)
print("terminated (pole fell):", terminated)
print("truncated (time limit):", truncated)
print("info:", info)

env.reset()
total_reward = 0
done = False
steps = 0

while not done and steps < 100:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    steps += 1

print("\nsteps survived:", steps)
print("total reward:", total_reward)
#  reward of +1 is given for every step taken, including the termination step

env.close()

print(
    """\nThe episode ends if any one of the following occurs:
\n1. Termination: Pole Angle is greater than ±12°
\n2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
\n3. Truncation: Episode length is greater than 500 for v1 which is this one\n"""
)
