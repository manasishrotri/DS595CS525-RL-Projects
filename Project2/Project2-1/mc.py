# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:24:03 2020

@author: manasi  shrotri
"""

import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise
    Parameters:
    -----------
    observation
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    score, dealer_score, usable_ace = observation
    # action
    action=1
    if(score>=20):
            action=0
        
    ############################
    return action

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.
    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    Note: at the begining of each episode, you need initialize the environment using env.reset()
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    for i in range(n_episodes):
        # initialize the episode
        this_state=env.reset()
        terminate=False
       
        # generate empty episode list
        episode=[]
        # loop until episode generation is done
        while( terminate == False):
            # select an action
            action=policy(this_state) 
            # return a reward and new state
            next_state,reward,terminate,_=env.step(action)
            # append state, action, reward to episode
            episode.append([this_state,action,reward])
            # update state to new state
            this_state=next_state
        print(episode)
        # loop for each step of episode, t = T-1, T-2,...,0
        
  		
		# Get total return for all states in episode
		# loop for each step of episode, t = T-1, T-2,...,0
        states_visited=[]
        for i,(St,action,reward) in enumerate(episode):
            #print(i,St)
            if St not in states_visited:
                returns_count[St]+=1
                g=reward
                print(g)
                p=1
                for k in range(i+1,len(episode)): 
                    s,a,r=episode[k]
                    g+=(gamma**p)*r
                    p+=1
                   # print(g)
                returns_sum[St]+=g
                V[St]= returns_sum[St]/returns_count[St]
                states_visited.append(St)
     ############################

    return V

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    policy=np.zeros(nA)
    
    policy[np.argmax(Q[state])]=1-epsilon
    policy=[x+(epsilon/nA) for x in policy]
    
    action = np.random.choice(np.arange(len(policy)), p = policy)
    ############################
    return action

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #

    # define decaying epsilon
   
        
    # initialize the episode
    for i in range(n_episodes):
        if epsilon <= 0.05:
            epsilon = 0.05
        else:
            epsilon = epsilon - (0.1/n_episodes)
        #if(epsilon>(0.1/n_episodes)):
        #    epsilon = epsilon-(0.1/n_episodes)

        # initialize the episode
        this_state=env.reset()
        terminate=False
        # generate empty episode list
        episode=[]
        # loop until one episode generation is done
        while( terminate == False):
            # get an action from epsilon greedy policy
            action=epsilon_greedy(Q, this_state, env.action_space.n, epsilon)
            # return a reward and new state
            next_state,reward,terminate,_=env.step(action)
            # append state, action, reward to episode
            episode.append([this_state,action,reward])
            # update state to new state
            this_state=next_state


        # loop for each step of episode, t = T-1, T-2, ...,0
        
        # compute G
        returns_G = defaultdict(list)
        G=0
        for St,action,reward in episode[::-1]:
            # compute G
            G=gamma*G+reward
            returns_G[St].append(G)
        
        stateactions_visited=[]    
        for St,action,reward in episode:
            SA = (St,action)
            # unless the pair state_t, action_t appears in <state action> pair list
            if SA not in stateactions_visited:
                # update return_count
                returns_count[SA]+=1
                # update return_sum
                
                returns_sum[SA]+=returns_G[St][-1]
                # calculate average return for this state over all sampled episodes
                Q[St][action]= returns_sum[SA]/returns_count[SA]
                stateactions_visited.append(SA)

    return Q
