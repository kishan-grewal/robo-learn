# dqn_cartpole/evaluate.py

import argparse
import torch
import gymnasium as gym
from train import QNetwork, state_to_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained CartPole-v1 DQN agent"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to run"
    )
    parser.add_argument(
        "--no-render",
        action="store_false",
        help="Render the environment with gymnasium",
    )
    args = parser.parse_args()

    # setup
    env = gym.make("CartPole-v1", render_mode="human" if args.no_render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # load network
    qnet = QNetwork(state_dim, action_dim)
    qnet.load_state_dict(torch.load("models_cartpole/cartpole_qnet.pth"))
    qnet.eval()

    # run episodes
    episode_rewards = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                q_values = qnet(state_to_tensor(obs))
                action = q_values.argmax().item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"Episodes {episode + 1}: {total_reward}")

    print(
        f"\nAverage reward over {args.episodes} episodes: {sum(episode_rewards) / len(episode_rewards):.2f}"
    )
    env.close()


if __name__ == "__main__":
    main()
