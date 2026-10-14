
import gymnasium as gym
import numpy as np

# Create the Lunar Lander game.
env = gym.make("LunarLander-v3", render_mode="human")

# Start one episode.
obs, info = env.reset(seed=42)

print("Observation:", obs)
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)

# Let the rocket take random actions.
for step in range(1000):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if step % 50 == 0:
        print(
            f"Step: {step}, "
            f"Reward: {reward:.2f}, "
            f"Observation: {np.round(obs, 2)}"
        )

    # Start another episode after the current one ends.
    if terminated or truncated:
        obs, info = env.reset()

env.close()
