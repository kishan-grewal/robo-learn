# ppo_gae_pendulum/evaluate.py

import argparse
import torch
import numpy as np
import gymnasium as gym
from train import ActorCritic, Config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Pendulum-v1 PPO+GAE agent"
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("Pendulum-v1", render_mode="human" if args.no_render else None)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_high = float(env.action_space.high[0])

    cfg = Config()

    # load network
    ac = ActorCritic(obs_dim, act_dim).to(device)
    ac.load_state_dict(
        torch.load("models_pendulum/pendulum_ppo_ac.pth", map_location=device)
    )
    ac.eval()

    # run episodes
    episode_rewards = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action_env, _, _, _ = ac.get_action_and_logprob(obs_t)
                action_np = action_env.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action_np)
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
