# ppo_vec_cheetah/evaluate.py

import argparse
import torch
import gymnasium as gym
from train_vectorised import ActorCritic, Config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained HalfCheetah-v5 PPO agent"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to run"
    )
    parser.add_argument(
        "--no-render",
        action="store_false",
        dest="render",
        help="Disable rendering",
    )
    args = parser.parse_args()

    # setup
    config = Config()
    env = gym.make("HalfCheetah-v5", render_mode="human" if args.render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # load actor-critic network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = ActorCritic(state_dim, action_dim, config).to(device)
    actor_critic.load_state_dict(torch.load("models_cheetah/cheetah_ppo_vec_ac.pth"))
    actor_critic.eval()

    # run episodes
    episode_rewards = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action, _ = actor_critic.get_action_and_logprob(obs_tensor)
                action = action.squeeze(0).cpu().numpy()

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
