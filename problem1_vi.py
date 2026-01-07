# Name: Elias Vitell
# Personal number: 0102057379

# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.


# 
# %%

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random


import problem1 as maze 

# %%
# Description of the maze as a numpy array
mazex = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]])
# With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze

env = maze.Maze(mazex) # Create an environment maze
horizon =  20      # TODO: Finite horizon
# horizon = 3

# env.show()

# Solve the MDP problem with dynamic programming
# V, policy = maze.dynamic_programming(env, horizon)  



# %% [markdown]
# Value iteration

# %%

# pickle was used during development but is not needed
# import pickle 

if True:
    V_vi, policy_vi = maze.value_iteration(env=env,gamma=0.9,epsilon=.1)


    # with open("data.pickle","rb") as f:
    #     pickle.dump(
    #         {
    #             "V":V_vi,
    #             "policy":policy_vi
    #         },f
    #     )
        

# %%

# with open("data.pickle","rb") as f:
#     data = pickle.load(f)
# V_vi = data["V"]
# policy_vi = data["policy"]

# %%
start  = ((0,0), (6,5))

path_vi = env.simulate(start, policy_vi, 'ValIter')[0]


# %%
maze.animate_solution(mazex,path_vi)

# %%
wins = 0
count = 0
for i in range(1000):
    path_vi = env.simulate(start, policy_vi, 'ValIter')[0]
    count +=1

    for p in path_vi:
        if type(p) is tuple:
            # just a normal position
            continue
        elif p == "Win":
            wins += 1
            break 
        else:
            break




# %%
print(float(wins)/float(count))


