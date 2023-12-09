
def get_default_success_rate(trajectories, goal=50):
    succ = 0
    tot = 0
    for traj in trajectories:
        if len(traj) > goal:
            succ += 1
        tot += 1
    return succ / tot
