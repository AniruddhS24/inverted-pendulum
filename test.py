# import gymnasium as gym
# env = gym.make("InvertedDoublePendulum-v4", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#     action = env.action_space.sample()  # this is where you would insert your policy
#     observation, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()

import data
import metrics

environment = data.Environment(render=False)
# environment.simulate(data.random_policy)

trajectories = [environment.sample_episode(
    data.random_policy) for _ in range(100)]
print(metrics.get_default_success_rate(trajectories))
