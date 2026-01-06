# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random
REWARDS = {
    "STEP_REWARD" : 1,
    "GOAL_REWARD" : 100000000,
    "IMPOSSIBLE_REWARD": -100000000000000000000,
    # "IMPOSSIBLE_REWARD": 0,
    "MINOTAUR_REWARD": -1,
}
# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
# BLACK        = '#000000'
BLACK        = '#000FFF' # so text can be seen. Should have a different name, but hopefully not to important
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values 
    STEP_REWARD = REWARDS["STEP_REWARD"]          #TODO
    GOAL_REWARD = REWARDS["GOAL_REWARD"]          #TODO
    IMPOSSIBLE_REWARD = REWARDS["IMPOSSIBLE_REWARD"]    #TODO
    MINOTAUR_REWARD = REWARDS["MINOTAUR_REWARD"]      #TODO

    def __init__(self, maze):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions

    def __states(self):
        
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            states[s] = ((i,j), (k,l))
                            map[((i,j), (k,l))] = s
                            s += 1
        
        states[s] = 'Eaten'
        map['Eaten'] = s
        s += 1
        
        states[s] = 'Win'
        map['Win'] = s
        
        return states, map

    def __move(self, state, action):               
        """ Makes a step in the maze, given a current position and an action. 
            If the action STAY or an inadmissible action is used, the player stays in place.
        
            :return list of tuples next_state: Possible states ((x,y), (x',y')) on the maze that the system can transition to.
        """
        
        if self.states[state] == 'Eaten' or self.states[state] == 'Win': # In these states, the game is over
            return [self.states[state]]
        
        else: # Compute the future possible positions given current (state, action)
            row_player = self.states[state][0][0] + self.actions[action][0] # Row of the player's next position 
            col_player = self.states[state][0][1] + self.actions[action][1] # Column of the player's next position 
            
            # Is the player getting out of the limits of the maze or hitting a wall?

            player_left = row_player < 0
            player_right = row_player >= self.maze.shape[0]
            player_above = col_player < 0
            player_below = col_player >= self.maze.shape[1]
            player_outside = player_left or player_right or player_above or player_below 
            if player_outside:
                impossible_action_player = True
            else:
                player_wall = self.maze[row_player, col_player] == 1
                impossible_action_player = player_wall
            
        
            actions_minotaur = [[0, -1], [0, 1], [-1, 0], [1, 0]] # Possible moves for the Minotaur
            rows_minotaur, cols_minotaur = [], []
            for i in range(len(actions_minotaur)):
                # Is the minotaur getting out of the limits of the maze?
                impossible_action_minotaur = (self.states[state][1][0] + actions_minotaur[i][0] == -1) or \
                                             (self.states[state][1][0] + actions_minotaur[i][0] == self.maze.shape[0]) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == -1) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == self.maze.shape[1])
            
                if not impossible_action_minotaur:
                    rows_minotaur.append(self.states[state][1][0] + actions_minotaur[i][0])
                    cols_minotaur.append(self.states[state][1][1] + actions_minotaur[i][1])  
          

            # Based on the impossiblity check return the next possible states.
            if impossible_action_player: # The action is not possible, so the player remains in place
                states = []
                for i in range(len(rows_minotaur)):
                    
                    if  (self.states[state][0][0] == rows_minotaur[i]) and (self.states[state][0][1] == cols_minotaur[i])          :                          # TODO: We met the minotaur
                        states.append('Eaten')
                    
                    elif self.maze[self.states[state][0][0], self.states[state][0][1]] ==2       :                           # TODO: We are at the exit state, without meeting the minotaur
                        states.append('Win')
                
                    else:     # The player remains in place, the minotaur moves randomly
                        states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i])))
                
                return states
          
            else: # The action is possible, the player and the minotaur both move
                states = []
                for i in range(len(rows_minotaur)):
                
                    if   (row_player == rows_minotaur[i]) and (col_player == cols_minotaur[i])         :                          # TODO: We met the minotaur
                        states.append('Eaten')
                    
                    elif self.maze[row_player,col_player] == 2       :                          # TODO:We are at the exit state, without meeting the minotaur
                        states.append('Win')
                    
                    else: # The player moves, the minotaur moves randomly
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i])))
              
                return states
        
        
        

    def __transitions(self) -> np.ndarray:
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # TODO: Compute the transition probabilities.
        for i in range(self.n_states):
            for k in range(self.n_actions):
                start_state = i
                action = k
                candidates = self.__move(start_state, action)
                state_count = len(candidates)
                prob = 1/state_count
                prev = []
                prev_map = {}
                for state in candidates:
                    so = self.map[state]
                    if so in prev:
                        if not (type(state) is str):
                            print("err", so)
                            print(state, "\n", prev_map[so])
                            raise Exception()
                    prev.append(so)
                    prev_map[so] = state 

                    # if not (type(prob) is float): print( start_state,so,action, "--", type(prob), prob)
                    
                    transition_probabilities[start_state, so, action] = prob

        
  
    
        # _ = input()
        return transition_probabilities



    def __rewards(self) -> np.ndarray:
        
        """ Computes the rewards for every state action pair """

        rewards = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                
                if self.states[s] == 'Eaten': # The player has been eaten
                    rewards[s, a] = self.MINOTAUR_REWARD
                
                elif self.states[s] == 'Win': # The player has won
                    rewards[s, a] = self.GOAL_REWARD
                
                else:                
                    next_states = self.__move(s,a)
                    next_s = next_states[0] # The reward does not depend on the next position of the minotaur, we just consider the first one
                    
                    if self.states[s][0] == next_s[0] and a != self.STAY: # The player hits a wall
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    
                    else: # Regular move
                        rewards[s, a] = self.STEP_REWARD

        return rewards




    def simulate(self, start, policy, method):
        
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        
        if method == 'DynProg':
            horizon = policy.shape[1] # Deduce the horizon from the policy shape
            t = 0 # Initialize current time
            s = self.map[start] # Initialize current state 
            path.append(start) # Add the starting position in the maze to the path
            
            while t < horizon - 1:
                a = policy[s, t] # Move to next state given the policy and the current state
                next_states = self.__move(s, a) 
                for s_cand in next_states:
                    print("Possible state:", s_cand)
                next_state_indices = [self.map[ns] for ns in next_states]
                # s_index = self.map[s]
                next_state_probabilities =[None]*len(next_state_indices)

                for i, nsi in enumerate(next_state_indices):
                    print(f"keys: s={s}, nsi={nsi}, a={a}")
                    next_state_probabilities[i]=self.transition_probabilities[ s ,nsi,a]
                # next_s = next_states[0] # FIXME
                print(f"total prob: {np.sum(next_state_probabilities)}")
                next_s = random.choices(next_states,weights=next_state_probabilities,k=1)[0]
                path.append(next_s) # Add the next state to the path
                t +=1 # Update time and state for next iteration
                s = self.map[next_s]
                
        if method == 'ValIter': 
            t = 1 # Initialize current state, next state and time
            s = self.map[start]
            path.append(start) # Add the starting position in the maze to the path
            next_states = self.__move(s, policy[s]) # Move to next state given the policy and the current state
            next_s = None # FIXME
            path.append(next_s) # Add the next state to the path
            
            horizon =    None # FIXME                           # Question e
            # Loop while state is not the goal state
            while s != next_s and t <= horizon:
                s = self.map[next_s] # Update state
                next_states = self.__move(s, policy[s]) # Move to next state given the policy and the current state
                next_s = None # FIXME
                path.append(next_s) # Add the next state to the path
                t += 1 # Update time for next iteration
        
        return [path, horizon] # Return the horizon as well, to plot the histograms for the VI



    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

        print('The transition probabilities:')
        print(self.transition_probabilities)


def dynamic_programming(env:Maze, horizon:int) -> tuple[np.ndarray, np.ndarray]:
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    #TODO
    theta = 0.1
    gamma = 1.0
    T = horizon
    t = T

    # t = min(3, t)
    state_count = env.n_states
    # print(f"S={state_count}")

    # these shall both be arbitrarily intilialised, so set to 0
    policy = np.zeros([state_count, T],dtype=int)
    V = np.zeros([state_count, T])

    # print("tp", env.transition_probabilities)


    current_state = env.map['Win']

    # max(r_T(s, a))
    # st
    # a in A_s
    # s in S

    # v = 0
    # delta = theta + 1
    # while delta > theta:
    #     delta = 0
    while t > 0:
        t_index = t -1
        for state in env.states:
            # v = V[state,t_index]
            V[state,t_index] = 0
            vmax = None
            amax = None # action for which v is vmax
            for action in env.actions:
                policy_action = action # policy[state,t]

                vtemp = 0 # the "candidates" wrt a
                reward = env.rewards[state,policy_action]
                vtemp += reward
                # since we don't have determinism, we can't just set next state. Rather, we have to loop through all next states
                for next_state in env.states:
                    # reward = env.rewards[state,policy_action]

                    # Expected to be 0 in most cases:
                    transition_probability = env.transition_probabilities[state,next_state,action]
                    if transition_probability <= 0:
                        continue
                    if t==T:
                        vtemp_plus = 0
                    else:
                        vtemp_plus = gamma * transition_probability * V[next_state, t_index+1]
                    vtemp += vtemp_plus
                    #print(f"r={reward}, tp={transition_probability}, Vnext={V[next_state,t]}, vtemp_plus={vtemp_plus} vtemp={vtemp}")
                
                if vmax is None:
                    # not yet properly initialised
                    vmax = vtemp
                    amax = action
                if vtemp > vmax:
                    vmax = vtemp
                    amax = action
                if not (type(action ) is int):
                    print(f"action {action} is not int, ({type(action)})")

            # assign V[] max for next_state, action
            # print(f"V[{state},{t}]={vmax}")
            # print(f"policy[{state},{t}]={amax}")
            V[state,t_index] = vmax
            policy[state,t_index] = amax
            # print(f"policy[{state},{t}]={policy[state,t_index]} ({type(policy[state,t_index])}), amax={amax} ({type(amax)})")
            if not (type(amax) is int):
                print(type(amax))
            
        t-=1
        print(f"t={t}")
        
    return V, policy
    for state in env.states:
        v_prev = v 
        v = 0
        for action in env.actions:
            transition_probiability = env.transition_probabilities[state,current_state,action]
            reward = env.rewards[state, action]

            v += policy[state][action] * (reward +0)

            if abs(transition_probiability) > 0:
                print(f"tp: {transition_probiability}, reward: {reward}")


    

def bad():
    sa_chain = [] 
    while t >= 0:
        # iterate, choose max
        max_prob = .0

        # assume stay if nothing better is ofund
        prev_state = state 
        prev_action = 0
        for state_candidate_key in env.states:
            state_candidate = env.states[state_candidate_key]
            # print(f"state_candidate: {state_candidate_key} -- {state_candidate}")

            for action in env.actions:
                # print("action:", type(action))

                # print( "t=",t, "key1:", type(state_candidate_key), state_candidate_key)
                # print( "t=",t,"key2:", type(state), state)
                # print( "t=",t,"key3:",type(action), action)
                # keys: current state, next state, action
                transition_probability = env.transition_probabilities[state_candidate_key,state,action]
                # print(f"transition_probability={transition_probability}")
                # sanity_check = not (state is None)
                is_max = transition_probability > max_prob
                not_same = state != state_candidate_key # how can this happen when action != 0?
                if not_same and (not is_max) and (transition_probability > .0):
                    print(">0:","t=",t ,state_candidate_key,state,action,transition_probability, "max=",max_prob)
                if is_max and not_same:
                    print( "t =",t, "found",type(state_candidate_key), state_candidate_key,"--", type(state),state,"--",action , "--" , transition_probability)
                    prev_state = state_candidate_key
                    prev_action = action
                    max_prob = transition_probability 
        # break
        
        sa_chain.append((prev_state,state,prev_action))
        state = prev_state
        # break
        # # print("t=",t)
        # env.actions
        # env.states
        # env.rewards
        # env.transition_probabilities

        t-=1
        max_prob = .0
    sa_chain = reversed(sa_chain)
    m_grid = [None] * 7
    for i in range(7):
        m_grid[i]=["*"] * 8
    i = 0
    for s,next_s, a in sa_chain:
        print(f"state: {s}, next state: {next_s}, action: {a}")
        pos_pm = env.states [state ]
        pos=pos_pm[0]
        print("pos", pos)

        if not (type(pos) is str):
            x=pos[0]
            y=pos[1]

            m_grid[x][y] = str(i)
        i+=1
    s = '\n'.join([  ''.join(row) for row in m_grid ])
    print("STEPS\n",s)





    return V, policy

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    #TODO




    return V, policy



def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -1: LIGHT_RED, -2: LIGHT_PURPLE}
    
    rows, cols = maze.shape # Size of the maze
    fig = plt.figure(1, figsize=(cols, rows)) # Create figure of the size of the maze

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create a table to color
    grid = plt.table(
        cellText = None, 
        cellColours = colored_maze, 
        cellLoc = 'center', 
        loc = (0,0), 
        edges = 'closed'
    )
    
    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    for i in range(0, len(path)):
        if path[i-1] != 'Eaten' and path[i-1] != 'Win':
            obj = grid.get_celld()[(path[i-1][0])]
            # print(type(obj))
            # _ = input()
            grid.get_celld()[(path[i-1][0])].set_facecolor(col_map[maze[path[i-1][0]]])
            # grid.get_celld()[(path[i-1][0])].get_text().set_text( grid.get_celld()[(path[i-1][0])].get_text() + "\n" + "p" + str(i))
            grid.get_celld()[(path[i-1][1])].set_facecolor(col_map[maze[path[i-1][1]]])
        if path[i] != 'Eaten' and path[i] != 'Win':
            # add a line indicating player (p) and time to the cell. Note that t=i+1
            pt0 = grid.get_celld()[(path[i][0])].get_text()
            pt0.set_text(pt0.get_text() + "\n" + "M" + str(i+1))

            grid.get_celld()[(path[i][0])].set_facecolor(col_map[-2]) # Position of the player

            # as above but "M" for minotaur
            pt1 = grid.get_celld()[(path[i][1])].get_text()
            pt1.set_text(pt1.get_text() + "\n" + "M" + str(i+1))

            grid.get_celld()[(path[i][1])].set_facecolor(col_map[-1]) # Position of the minotaur
        display.display(fig)
        time.sleep(0.1)
        display.clear_output(wait = True)



if __name__ == "__main__":
    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])
    # With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze
    
    env = Maze(maze) # Create an environment maze
    horizon =  20      # TODO: Finite horizon

    env.show()

    # Solve the MDP problem with dynamic programming
    V, policy = dynamic_programming(env, horizon)  

    # Simulate the shortest path starting from position A
    method = 'DynProg'
    start  = ((0,0), (6,5))
    path = env.simulate(start, policy, method)[0]

    animate_solution(maze, path)