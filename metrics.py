
def get_metrics(trajectories, goal=50):
    succ = 0
    tot = 0
    avg_len = 0
    for traj in trajectories:
        avg_len += len(traj)
        if len(traj) > goal:
            succ += 1
        tot += 1
    return succ / tot, avg_len / tot
