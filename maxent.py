import argparse
import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from data import Environment, make_env
import metrics
import json


class MaxentRewardModel():
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.theta = np.random.uniform(-1, 1, size=n_features)

    def get_reward(self, x):
        return np.dot(self.theta, x)

    def update_model(self, expert, learner, lr=1e-3):
        self.theta += lr*(expert - learner)
        self.theta = np.clip(self.theta, -1, 1)

    def expert_feature_expectations(self, feature_matrix, expert_demos):
        freq = np.zeros((self.n_features))
        for trajectory in expert_demos:
            for state in trajectory:
                freq += feature_matrix[state]
        return freq / len(expert_demos)


class PolicyGradient(nn.Module):
    def __init__(self, n_features, n_actions):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.mu = nn.Linear(n_features, n_actions)
        self.std = nn.Linear(n_features, n_actions)

        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.xavier_uniform_(self.std.weight)

    def forward(self, x):
        mu = self.mu(x)
        theta = torch.exp(self.std(x))
        return mu, theta

    def policy(self, x):
        mu, theta = self.forward(x)
        return Normal(mu, theta).sample().item()


def train(batch_size, pg_lr, reward_lr, n_episodes, n_discrete_split):
    env = Environment(render=False, seed=1)
    n_states = n_discrete_split**7
    n_actions = 1
    feature_matrix = np.eye(n_states, dtype=np.float32)

    def get_state(obs):
        obs = obs[:7]
        env_low = env.env_low[:7]
        env_high = env.env_high[:7]
        env_diff = (env_high - env_low) / n_discrete_split
        state = np.floor((obs - env_low) / env_diff).astype(int)
        state_idx = 0
        for i in range(len(state)):
            state_idx += state[i] * (n_discrete_split ** i)
        return int(state_idx)

    def trajectories_to_states(trajectories):
        states = [[0 for _ in range(len(trajectories[0]))]
                  for _ in range(len(trajectories))]
        for i, traj in enumerate(trajectories):
            for j, t in enumerate(traj):
                states[i][j] = get_state(t['state'])
        return states

    reward_model = MaxentRewardModel(n_features=n_states)
    pg_model = PolicyGradient(n_features=n_states, n_actions=n_actions)
    optimizer = optim.Adam(pg_model.parameters(), lr=pg_lr)
    learner_exp = np.zeros(n_states)

    # later replace with actual expert trajectories
    expert_demos = [env.sample_episode(
        lambda x: 0, full_trajectory=True) for _ in range(100)]
    expert_demos = trajectories_to_states(expert_demos)
    expert = reward_model.expert_feature_expectations(
        feature_matrix, expert_demos)

    scores = []
    accum_loss = 0

    print(
        f'Training... | Batch Size: {batch_size} | States: {n_states} | Actions: {n_actions}')
    for episode in range(n_episodes):
        if episode >= 1000 and episode % 100 == 0:
            learner = learner_exp / episode
            reward_model.update_model(expert, learner, lr=reward_lr)

        # Sample trajectory
        observation = env.reset()
        score = 0
        steps = 0
        states = []
        actions = []
        irl_rewards = []
        while True:
            state = get_state(observation)
            action = pg_model.policy(torch.from_numpy(feature_matrix[state]))
            next_state, reward, done, _, _ = env.step(action)
            score += reward
            irl_reward = reward_model.get_reward(feature_matrix[state])

            states.append(state)
            actions.append(action)
            irl_rewards.append(irl_reward)
            steps += 1

            learner_exp += feature_matrix[state, :]
            score += reward
            observation = next_state

            if done:
                break
        scores.append(score/steps)

        # Discount and normalize rewards
        for i in reversed(range(steps)):
            if i == steps - 1:
                continue
            irl_rewards[i] = irl_rewards[i] + 0.99 * irl_rewards[i + 1]
        irl_rewards = np.array(irl_rewards)
        irl_rewards = (irl_rewards - np.mean(irl_rewards))/np.std(irl_rewards)

        # Calculate loss
        loss = 0
        for i in range(steps):
            state = states[i]
            action = actions[i]
            irl_reward = irl_rewards[i]
            mu, theta = pg_model(torch.from_numpy(feature_matrix[state]))
            log_prob = Normal(mu, theta).log_prob(torch.Tensor([action]))
            loss += -log_prob * irl_reward
        accum_loss += loss/steps

        # Policy gradient update
        if episode % batch_size == 0:
            optimizer.zero_grad()
            accum_loss.backward()
            optimizer.step()
            print(
                f'Episode {episode} | Avg Score: {np.mean(scores)} | Avg Loss: {accum_loss.item()}')
            accum_loss = 0


train(32, 1e-3, 1e-3, 10000, 4)
