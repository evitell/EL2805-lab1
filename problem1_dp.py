# Name: Elias Vitell
# Personal number: 0102057379

# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.


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

# Simulate the shortest path starting from position A
method = 'DynProg'
start  = ((0,0), (6,5))




# %%

# Solve the MDP problem with dynamic programming
V, policy = maze.dynamic_programming(env, 20)  


# %%
# def outcome(path, maxlen=-1):
#     if maxlen==-1:
#         maxlen=len(path)
#     for i,p in enumerate(path):
#         if p=="Win":
#             return 1 
#         if p=="Eaten":
#             return -1 
#         if  i+1 == maxlen:
#             break
#     if not (i+1==maxlen):
#         raise Exception("index error")
#     return 0




# %%

path = env.simulate(start, policy, method)[0]
maze.animate_solution(mazex, path)


# %%
h_data=[]
w_data=[]
Hmax = 30
# Hmax = 17
V30, policy30 = maze.dynamic_programming(env, Hmax)

for horizon in range(1,Hmax +1):
    # V, policy = maze.dynamic_programming(env, horizon)
    wins = 0
    count = 0
    policy_h = policy30[:,Hmax-horizon:]
    assert(policy_h.shape[1]==horizon)
    for i in range(100):

        path = env.simulate(start, policy_h, method)[0]
        # res = outcome(path,i)
        for pi in path:#[:horizon]:
            if pi =="Win":
                wins +=1 
                break
        count += 1
    
    win_percentage = float(wins)/float(count)
    h_data.append(horizon)
    w_data.append(win_percentage)


# plt.plot(h_data,w_data)
# plt.show()
    

# %%

plt.xlabel("hirozon")
plt.ylabel("Win percentage")
plt.plot(h_data,w_data,label="Win percentage")
plt.legend()
plt.title("Win percentage as function of time horizon\nwhen the minotaur must move")
plt.show()

# %%
env_still = maze.Maze(mazex,True)
V30_still, policy30_still = maze.dynamic_programming(env_still, Hmax)


# %%
h_data_still =[]
w_data_still =[]
for horizon in range(1,Hmax +1):
    # V, policy = maze.dynamic_programming(env, horizon)
    wins = 0
    count = 0
    for i in range(100):
        path = env_still.simulate(start, policy30_still, method)[0]
        # res = outcome(path,i)
        for pi in path[:horizon]:
            if pi =="Win":
                wins +=1 
                break
        count += 1
    
    win_percentage = float(wins)/float(count)
    h_data_still.append(horizon)
    w_data_still.append(win_percentage)


# plt.plot(h_data,w_data)
# plt.show()
    

# %%

plt.xlabel("hirozon")
plt.ylabel("Win percentage")
plt.plot(h_data_still,w_data_still,label="Win percentage")
plt.legend()
plt.title("Win percentage as function of time horizon\nwhen the minotaur may stand still")
plt.show()


