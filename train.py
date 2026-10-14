
from pathlib import Path
import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from reward_wrapper import RewardExperimentWrapper


# Read the experiment name from the command line.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment",
    choices=["baseline", "shaped", "hack"],
    default="baseline"
)
parser.add_argument(
    "--timesteps",
    type=int,
    default=100_000,
    help="Number of training timesteps",
)
args = parser.parse_args()

experiment = args.experiment

# Create folders.
Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)
Path("logs/tensorboard").mkdir(parents=True, exist_ok=True)

# Create the original environment.
env = gym.make("LunarLander-v3")

# Apply the appropriate reward configuration.
if experiment == "shaped":
    env = RewardExperimentWrapper(
        env,
        gamma=0.99,
        use_progress_reward=True,
        time_penalty=0.01
    )

elif experiment == "hack":
    env = RewardExperimentWrapper(
        env,
        use_progress_reward=False,
        time_penalty=0.0,
        upright_slow_bonus=1.0
    )

# Monitor records episode rewards and lengths.
env = Monitor(
    env,
    filename=f"logs/{experiment}.csv"
)

# Create the PPO agent.
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    policy_kwargs={"net_arch": [64, 64]},
    verbose=1,
    seed=42,
    tensorboard_log="logs/tensorboard"
)

# Train the selected experiment.
model.learn(
    total_timesteps=args.timesteps,
    tb_log_name=experiment
)

# Save a separate model for each experiment.
model_path = f"models/ppo_{experiment}"
model.save(model_path)

env.close()

print(f"Training finished: {experiment}")
print(f"Model saved at: {model_path}")
