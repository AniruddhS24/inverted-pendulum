
def get_metrics(trajectories, goal=50):
    succ = 0
    tot = 0
    avg_len = 0
    # total_reward = 0
    for traj in trajectories:
        avg_len += len(traj)
        if len(traj) > goal:
            succ += 1
        tot += 1
        # total_reward += sum(step["reward"][0] for step in traj)
    return succ / tot, avg_len / tot

def get_reward_total(trajectories):
    total_reward = 0
    for traj in trajectories:
        total_reward += sum(step["reward"] for step in traj)
    return total_reward