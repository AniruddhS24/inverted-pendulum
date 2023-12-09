import gymnasium as gym
import numpy as np


class Environment:
    def __init__(self, render=False):
        self.render = render
        if render:
            self.env = gym.make("InvertedDoublePendulum-v4",
                                render_mode="human")
        else:
            self.env = gym.make("InvertedDoublePendulum-v4")

    def reset(self, seed):
        observation, info = self.env.reset(seed=seed)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        return observation, reward, terminated, truncated, info

    def simulate(self, policy):
        if not self.render:
            raise Exception(
                "Cannot simulate environment without render mode enabled.")
        observation, _ = self.reset(seed=42)
        for _ in range(1000):
            action = policy(observation)
            observation, reward, terminated, truncated, info = self.step(
                action)
            if truncated:
                observation, info = self.reset(seed=42)

    def sample_episode(self, policy, max_steps=500):
        trajectory = []
        observation, info = self.reset(seed=42)
        trajectory.append({'timestep': 0,
                           'state': observation,
                           'action': 0,
                          'reward': 0})
        for t in range(1, max_steps):
            action = policy(observation)
            observation, reward, terminated, truncated, info = self.step(
                action)
            trajectory.append({
                'timestep': t,
                'state': observation,
                'action': action,
                'reward': reward
            })
            if terminated or truncated:
                return trajectory
        return trajectory


def random_policy(observation):
    return np.random.uniform(-1, 1, size=1)


# def policy(obs):
#     return action
