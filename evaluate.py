
import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def evaluate_model(model_path, episodes=20, seed=1000, render=False):
    # Load the trained PPO model.
    print(f"\nLoading model: {model_path}")

    model = PPO.load(model_path)

    # Use the original environment for a fair comparison.
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    episode_results = []

    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)

            total_reward = 0.0
            terminated = False
            truncated = False

            final_step_reward = None
            while not (terminated or truncated):
                # Deterministic actions make evaluation more consistent.
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, terminated, truncated, info = env.step(action)
                final_step_reward = float(reward)
                total_reward += final_step_reward

                # total_reward += float(reward)

            # Inspect the terminal state and environment outcome.
            final_obs = obs

            both_legs_contact = bool(
                final_obs[6] > 0.5 and final_obs[7] > 0.5
            )

            # Gymnasium LunarLander uses game_over to indicate
            # a crash or certain failure conditions.
            game_over = getattr(env.unwrapped, "game_over", None)

            # if game_over is not None:
            #     landed = bool(
            #             terminated
            #             and not truncated
            #             and final_step_reward is not None
            #             and final_step_reward > 0
            #     )
            #     landing_method = "terminal_reward_sign"
            # else:
            #     # Fallback only: this is a heuristic, not a guarantee.
            #     speed = np.hypot(final_obs[2], final_obs[3])
            #     stable_pose = (
            #         both_legs_contact
            #         and speed < 0.2
            #         and abs(final_obs[4]) < 0.2
            #     )
            #     landed = bool(
            #         terminated and not truncated and stable_pose
            #     )
            #     landing_method = "terminal_state_heuristic"
            landed = bool(
                terminated
                and not truncated
                and final_step_reward is not None
                and final_step_reward > 0
            )

            landing_method = "terminal_reward_sign"
            score_200 = total_reward >= 200

            episode_results.append({
                "episode": episode + 1,
                "seed": seed + episode,
                "reward": total_reward,
                "score_at_least_200": score_200,
                "landed": landed,
                "both_legs_contact": both_legs_contact,
                "landing_method": landing_method,
            })

            print(
                f"Episode {episode + 1:02d}/{episodes} | "
                f"Reward: {total_reward:8.2f} | "
                f"Score >= 200: {score_200} | "
                f"Landed: {landed}"
            )

    finally:
        env.close()

    # Calculate evaluation metrics.
    rewards = [row["reward"] for row in episode_results]
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    score_count = sum(row["score_at_least_200"] for row in episode_results)
    landing_count = sum(row["landed"] for row in episode_results)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Model: {model_path}")
    print(f"Episodes evaluated: {episodes}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Reward standard deviation: {std_reward:.2f}")
    print(f"Episodes scoring >= 200: {score_count}/{episodes}")
    print(f"Landing successes detected: {landing_count}/{episodes}")
    print(f"Score-threshold rate: {score_count / episodes:.1%}")
    print(f"Detected landing rate: {landing_count / episodes:.1%}")

    # Save individual episode results for later analysis and plotting.
    Path("results").mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).name
    output_file = Path("results") / f"evaluation_{model_name}.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=episode_results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(episode_results)

    print(f"Episode results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Lunar Lander PPO model."
    )

    parser.add_argument(
        "--model",
        default="models/ppo_baseline",
        help="Path to the saved PPO model",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Starting random seed",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Display the lander while evaluating",
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    evaluate_model(
        model_path=args.model,
        episodes=args.episodes,
        seed=args.seed,
        render=args.render,
    )


if __name__ == "__main__":
    main()
