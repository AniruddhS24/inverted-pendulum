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
        # self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.env = gym.wrappers.ClipAction(self.env)
        self.env = gym.wrappers.NormalizeObservation(self.env)
        self.env = gym.wrappers.TransformObservation(
            self.env, lambda obs: np.clip(obs, -10, 10))
        self.env = gym.wrappers.NormalizeReward(self.env)
        self.env = gym.wrappers.TransformReward(
            self.env, lambda reward: np.clip(reward, -10, 10))

    def observation_space(self):
        return self.env.observation_space.shape

    def action_space(self):
        return self.env.action_space.shape

    def reset(self, seed):
        observation, _ = self.env.reset(seed=seed)
        return observation

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
        observation = self.reset(seed=42)
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


def make_env():
    def thunk():
        return Environment(render=False).env
    return thunk


def random_policy(observation):
    return np.random.uniform(-1, 1, size=1)


# def policy(obs):
#     return action
