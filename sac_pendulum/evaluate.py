# sac_pendulum/evaluate.py

import argparse
import torch
import gymnasium as gym
from train import Actor, state_to_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Pendulum-v1 SAC agent"
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
    env = gym.make("Pendulum-v1", render_mode="human" if args.no_render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # load actor network
    actor = Actor(state_dim, action_dim)
    actor.load_state_dict(torch.load("models_pendulum/pendulum_sac_actor.pth"))
    actor.eval()

    # run episodes
    episode_rewards = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action, _ = actor.sample(state_to_tensor(obs))
                action = action.squeeze(0).numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: {total_reward:.2f}")

    print(
        f"\nAverage reward over {args.episodes} episodes: {sum(episode_rewards) / len(episode_rewards):.2f}"
    )
    env.close()


if __name__ == "__main__":
    main()
