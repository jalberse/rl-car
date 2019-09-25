import gym
import numpy as np
import cv2 as cv
import sys
from preprocessing import rgb_to_bw_threshold
import random
from collections import defaultdict

np.set_printoptions(threshold=sys.maxsize)

ACTION_SPACE_SIZE = 9 # Number of possible actions

def e_greedy(Q,state,epsilon):
    """
    Returns an action key given Q and state
    An action key in {0,..,ACTION_SPACE_SIZE-1}, which is mapped to a possible action in T x G x B
        Returns the optimal action given Q, s with a probability of (1-epsilon)
        Returns a random action with a probability of epsilon
    """
    # TODO real implementation
    if (random.random() < 1-epsilon):
        return np.argmax(Q[state])
    else:
        return random.randint(0,ACTION_SPACE_SIZE-1)

# TODO: Each episode Terminating after 1000 timesteps rather than just when 
#       done or exiting playfield. Why?
#       Our local version of car_racing.py might differ from what
#       I'm actually importing. Check out that code and how it's setting done.
#       Consider though: Is it best to end each episode afte 1000 steps after all?
#       It will still learn to move forward and may cut down on episode lengths
#       at beginning of training
# TODO: Because we don't know how many episodes are realistic to train, try 
#       plotting every n episodes so we at least have some plots if we can't
#       finish the training run. Maybe a key to stop training and return Q immediately
#       so we can save it at any time.
def q_learning_train(env, num_episodes, discount_rate = 0.99, learning_rate = 0.01, epsilon = 0.05):
    """
    Note gamma is discount_rate, alpha is learning_rate

    Returns Q(s,a), the value for all state, action pairs in S x A, after training with epsilon-greedy policy
    """
    # Initialize Q(s,a) for all s in S, a in A (mathematically)
    # In reality we initialize as each state S comes in (many states are unreachable, so don't waste time)
    
    # Action space is [steer, gas, brake] with bounds [[-1,1],[0,1],[0,1]] in real numbers
    # We choose to discretize s.t.
    # steer is in T = {-1, 0, 1} (left, straight, right)
    # gas is in   G = {0,1} (gas or no gas)
    # and brake   B = {0, 1} (brake or no brake)
    # This discretization should be OK - it is how controls for humans are discretized, after all
    #   Note: I think some literature on this environ also discretizes in this way. Check.
    # Action space is therefore T x G x B less situations where we have both brake and gas active (9 possible options)
    Q = defaultdict(lambda: np.zeros(ACTION_SPACE_SIZE)) # So Q[state] is a list of 12 values - a value for each of 12 actions in the state

    # Maps action_key index to action the env can understand
    actions = [
        [0.,0.,0.],
        [0.,0.,1.],
        [0.,1.,0.],
        [-1.,0.,0.],
        [-1.,0.,1.],
        [-1.,1.,0.],
        [1.,0.,0.],
        [1.,0.,1.],
        [1.,1.,0.],
    ]

    # Keep track of information we want to plot later
    statistics = dict([
        ('rewards',np.zeros(num_episodes)), # Total reward obtained this episode
        ('lap_time',np.zeros(num_episodes)), # Timesteps taken in episode
    ])

    for episode_cnt in range(num_episodes):
        observation = env.reset()
        t = 0
        # Get the initial state
        state = rgb_to_bw_threshold(observation) # state is a flattened 1-D tuple (hashable) from original np array
        # Game loop
        while(True):
            t += 1
            env.render() # TODO: (maybe) Delete for speed

            # Take action a, observe r, s'
            action_key = e_greedy(Q,state,epsilon)
            observation, reward, done, info = env.step(actions[action_key])
            new_state = rgb_to_bw_threshold(observation)

            # Q(s,a) <- Q(s,a) + alpha[r + gamma*max_a'(Q(s',a')) - Q(s,a)]
            Q[state][action_key] = Q[state][action_key] + learning_rate*(reward + discount_rate*Q[new_state][np.argmax(Q[new_state])] - Q[state][action_key])
            
            # s <- s'
            state = new_state

            # TODO update statistics dict
            # TODO plot as we go? save Q as we go?

            # TODO Incrementally save the state in case training stops. Save Q and statistics in readable format.
            #       This will also let us run the algo with a Q at a given snapshot and see it in action

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    return Q, statistics

if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    episodes = 10000
    discount_rate = 0.99
    learning_rate = 0.01
    epsilon = 0.05

    # Train the model
    Q, statistics = q_learning_train(env,episodes,discount_rate,learning_rate,epsilon)

    env.close()

    # TODO make some plots