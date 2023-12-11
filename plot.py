import json
import matplotlib.pyplot as plt
import math
from data import Environment
import metrics
import torch

niterations = 250
windowlen = 1

datafile = open("./checkpoints/ppo_5_250.json", "r")
fullstr = datafile.read()
decoded = json.JSONDecoder().decode(fullstr)

statlist = decoded["stats"]
avgrunlist = [row["avgruns"] for row in statlist]

# rewardlist = decoded["unshaped_rewards"]
# s = sum(rewardlist)
# print(s)
# count = 0
# for i in rewardlist:
#     if i > 5:
#         count += 1
# print(count)
# print(s/decoded["trajectories"])
# exit(0)

idxs = [i+1 for i in range(niterations)]


# print(avgrunlist)
avg = sum(avgrunlist[:windowlen])/windowlen
movingaverage = [avg]
for i in range(windowlen, niterations):
    avg -= avgrunlist[i-windowlen]/windowlen
    avg += avgrunlist[i]/windowlen
    movingaverage.append(avg)

acclist = [row["acc"] for row in statlist]
plt.scatter([i+1 for i in range(len(acclist))], acclist)
# plt.scatter([i+1 for i in range(len(movingaverage))], movingaverage)
plt.xlabel("No reward shaping")
plt.ylabel("Accuracy")
plt.show()
