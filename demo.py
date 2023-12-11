import torch
import gymnasium as gym
from ppo import PPOAgent
from data import Environment, make_env
import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

envs = gym.vector.SyncVectorEnv(
    [make_env()
     for i in range(1)]
)
model = PPOAgent(envs).to(device)

model.load_state_dict(torch.load("./checkpoints/ppo_null_750.pt"))

# these lines are for making a video

# demo_env = Environment(render=True)

# for i in range(100):
#     demo_env.sample_episode(model.policy)

def evaluate(ppo, num_episodes=100):
    env = Environment(render=False, demo=True)
    tot = 0
    for i in range(num_episodes):
        # print(i)
        traj = env.sample_episode(ppo.policy)
        tot += metrics.get_reward_total([traj])
    return tot / num_episodes

print(evaluate(model))