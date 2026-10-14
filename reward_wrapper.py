
import gymnasium as gym
import numpy as np


class RewardExperimentWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        gamma=0.99,
        use_progress_reward=False,
        time_penalty=0.0,
        upright_slow_bonus=0.0,
    ):
        super().__init__(env)

        self.gamma = gamma
        self.use_progress_reward = use_progress_reward
        self.time_penalty = time_penalty
        self.upright_slow_bonus = upright_slow_bonus

        self.previous_potential = 0.0

    def potential(self, obs):
        x, y, vx, vy, angle, angular_velocity, left, right = obs

        distance = np.hypot(x, y)
        speed = np.hypot(vx, vy)

        return -(
            distance
            + 0.5 * speed
            + 0.3 * abs(angle)
            + 0.1 * abs(angular_velocity)
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.previous_potential = self.potential(obs)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        current_potential = self.potential(obs)

        # Reward progress, rather than merely staying in one state.
        if self.use_progress_reward:
            reward += (
                self.gamma * current_potential
                - self.previous_potential
            )

        # Small penalty for taking another step.
        reward -= self.time_penalty

        # Deliberately flawed bonus for the reward-hacking experiment.
        speed = np.hypot(obs[2], obs[3])

        if abs(obs[4]) < 0.1 and speed < 0.1:
            reward += self.upright_slow_bonus

        self.previous_potential = current_potential

        return obs, reward, terminated, truncated, info
