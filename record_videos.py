
import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO


MODELS = {
    "baseline": "models/ppo_baseline",
    "shaped": "models/ppo_shaped",
    "hack": "models/ppo_hack",
}


def record_model(name, model_path, episodes=2, seed=2000):
    print(f"\nRecording model: {name}")

    # Verify that the trained model exists.
    if not Path(f"{model_path}.zip").exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}.zip"
        )

    model = PPO.load(model_path)

    # Store each experiment's videos separately.
    video_dir = Path("videos") / name
    video_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "LunarLander-v3",
        render_mode="rgb_array",
    )

    env = RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda episode_id: episode_id < episodes,
        name_prefix=name,
        disable_logger=True,
    )

    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)

            terminated = False
            truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(reward)

            print(
                f"{name} | Episode {episode + 1}/{episodes} "
                f"| Reward: {total_reward:.2f}"
            )

    finally:
        # Close the environment to finalize the recordings.
        env.close()

    print(f"Videos saved in: {video_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Record videos of the trained Lunar Lander agents."
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of episodes to record per model",
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    for name, model_path in MODELS.items():
        record_model(
            name=name,
            model_path=model_path,
            episodes=args.episodes,
        )


if __name__ == "__main__":
    main()
