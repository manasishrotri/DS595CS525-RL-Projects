# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:26:48 2020
RL Project 1: Frozen Lake MDP
@author: Manasi Shrotri
"""
### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:
	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.
    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #
    def next_state_reward(P,state,action,gamma,value_function):
        sum_reward=0
        for p,nextS,r,boolean_v in P[state][action]:
           sum_reward+=p*( r + gamma* value_function[nextS])
    #print(sum_reward)    
        return sum_reward

    while True:
        delta=0;
        for state in range(nS):
            new_value=0;
            for action in range(nA):
                sum_reward=next_state_reward(P,state,action,gamma,value_function)
                new_value+=policy[state][action]*sum_reward
            delta= max(delta, abs(new_value-value_function[state]))
            value_function[state] = new_value
        #print(value_function)
        if(delta < tol):
                break

    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.
    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.ones([nS, nA]) / nA
	############################
	# YOUR IMPLEMENTATION HERE #
    #iteration_policy=new_policy
    for state in range(nS):
        #current_policy=new_policy[state] 
        action_policy = np.zeros(nA)    
        for action in range(nA):
                 for p,nextS,r,boolean_v in P[state][action]:
                     action_policy[action] += p*( r + gamma* value_from_policy[nextS])
                     #print(action_policy)
        updated_policy=np.zeros(nA)
        updated_policy[np.argmax(action_policy)]= 1
        #print(updated_policy)            
        new_policy[state]=updated_policy
    
 	############################
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.
    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy_iter = policy.copy()
	############################
	# YOUR IMPLEMENTATION HERE #
    while True:
       current_policy=new_policy_iter.copy()
       Val_iter = policy_evaluation(P, nS, nA, new_policy_iter, gamma=0.9, tol=1e-8)
       new_policy_iter= policy_improvement(P, nS, nA, Val_iter, gamma=0.9)
       
       if np.array_equal(current_policy, new_policy_iter):
           break            
	############################
    return new_policy_iter, Val_iter

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.
    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    
    V_new = V.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        deltaV=0;
        for state in range(nS):
           # new_value=0;
            action_policy = np.zeros(nA)    
            for action in range(nA):
                 for p,nextS,r,boolean_v in P[state][action]:
                     action_policy[action] += p*( r + gamma* V[nextS])
            V_new[state]=max(action_policy)
            deltaV= max(deltaV, abs(V_new[state]-V[state]))
            V[state] = V_new[state]
        #print(value_function)
        if(deltaV < tol):
                break
        policy_new = policy_improvement(P, nS, nA, V_new, gamma)
    ############################
    return policy_new, V_new

def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.
    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob = env.reset() # initialize the episode
        done = False
        while not done:
            if render:
                env.render() # render the game
            ############################
            # YOUR IMPLEMENTATION HERE #
            #env.step(np.where(policy[0]==1)[0].tolist()[0])
            agent_next_step=env.step(np.argmax(policy[ob,:]))
            ob=agent_next_step[0]
            reward= agent_next_step[1]
            done= agent_next_step[2]
            total_rewards+=reward
            if done:
                break
    return total_rewards
