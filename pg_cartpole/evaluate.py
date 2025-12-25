# dqn_cartpole/evaluate.py

import argparse
import torch
import gymnasium as gym
from train import PolicyNetwork, ValueNetwork, state_to_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained CartPole-v1 PG agent"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to run"
    )
    parser.add_argument(
        "--render", action="store_false", help="Render the environment with gymnasium"
    )
    args = parser.parse_args()

    # setup
    env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # load policy network
    policy_net = PolicyNetwork(state_dim, action_dim)
    policy_net.load_state_dict(torch.load("models/cartpole_pg_policy.pth"))
    policy_net.eval()

    # run episodes
    episode_rewards = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                logits = policy_net(state_to_tensor(obs))
                action = logits.argmax(dim=1).item()

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
