import gym
import numpy as np
import cv2 as cv
import sys
from preprocessing import rgb_to_bw_threshold

from collections import defaultdict

np.set_printoptions(threshold=sys.maxsize)

def e_greedy(Q,state,epsilon):
    """
    Returns an action given Q and state
        Returns the optimal action given Q, s with a probability of (1-epsilon)
        Returns a random action with a probability of epsilon
    """

    pass

# TODO: Map action indices (of 12) to actions we can pass to env (i.e. [steer,gas,break] floats)

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

    Returns Q(s,a), the value for all state, action pairs in S x A
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
    # Action space is therefore T x G x B (12 possible options)
    Q = defaultdict(lambda: np.zeros(12)) # So Q[state] is a list of 12 values - a value for each of 12 actions in the state

    # Keep track of information we want to plot later
    statistics = dict([
        ('rewards',np.zeros(num_episodes)), # Total reward obtained this episode
        ('lap_time',np.zeros(num_episodes)), # Timesteps taken in episode
    ])

    for episode_cnt in range(num_episodes):
        observation = env.reset()
        t = 0
        # Get the initial state
        state = rgb_to_bw_threshold(observation)
        # Game loop
        while(True):
            t += 1
            env.render() # TODO: (maybe) Delete for speed

            # Take action a, observe r, s'
            action = env.action_space.sample() # TODO implement e-greedy not random
            observation, reward, done, info = env.step(action)
            new_state = rgb_to_bw_threshold(observation)

            cv.imshow("State",state) # TODO delete for speed
            cv.waitKey(1)

            # TODO
            # Q(s,a) <- Q(s,a) + alpha[r + gamma*max_a'(Q(s',a')) - Q(s,a)]
            
            # s <- s'
            state = new_state

            # TODO update statistics dict
            # TODO plot as we go? save Q as we go?

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