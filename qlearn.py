import gym
import numpy as np
import cv2 as cv
import sys
import os
import datetime
import json
import pickle
from preprocessing import rgb_to_bw_threshold
import random
import math
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
    if (random.random() < 1-epsilon):
        return np.argmax(Q[state])
    else:
        return random.randint(0,ACTION_SPACE_SIZE-1)

def default_action_generator():
    # Replaces typical lambda for defaultdict so we may pickle it
    return np.zeros(ACTION_SPACE_SIZE)

# TODO  reduce epsilon over time?
def q_learning_train(env, 
                     num_episodes, 
                     snapshots_dir, 
                     discount_rate = 0.99, 
                     learning_rate = 0.01, 
                     epsilon = 0.05, 
                     epsilon_decay = .99, 
                     epsilon_floor = 0, 
                     snapshot_freq=1000):
    """
    Returns Q(s,a), the value for all state, action pairs in S x A, after training with Q-learning with an epsilon-greedy policy

    env                 openAI gym environment (CarRacing-v0)
    num_episodes        The number of episodes to train. TODO -1 to continue indefinately until we ctrl+c
    discount_rate       gamma
    learning_rate       alpha
    epsilon             used in epsilon-greedy policy.
    epsilon_decay       decay per episode
    epsilon_floor       epsilon will not decay below this number
    run_id              used to identify the run. Creates a snapshots folder using this id to store progress
    snapshot_freq       Save Q, statistics every snapshot_freq episodes
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
    Q = defaultdict(default_action_generator) # So Q[state] is a list of 12 values - a value for each of 12 actions in the state

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
        ('lap_times',np.zeros(num_episodes)), # Timesteps taken in episode
    ])

    for episode_cnt in range(num_episodes):
        print(f'Episode: {episode_cnt}')
        if (epsilon > epsilon_floor):
            epsilon = epsilon * epsilon_decay
        print(f'Epsilon: {epsilon}')
        observation = env.reset()
        t = 0
        reward_total = 0
        # Get the initial state
        state = rgb_to_bw_threshold(observation) # State is bitpacked into a uint8 tuple
        # Game loop
        while(True):
            t += 1
            env.render() # TODO uncomment/comment to render/not render racing at every step

            # Take action a, observe r, s'
            action_key = e_greedy(Q,state,epsilon)
            observation, reward, done, info = env.step(actions[action_key])
            new_state = rgb_to_bw_threshold(observation)

            # Q(s,a) <- Q(s,a) + alpha[r + gamma*max_a'(Q(s',a')) - Q(s,a)]
            Q[state][action_key] = Q[state][action_key] + learning_rate*(reward + discount_rate*Q[new_state][np.argmax(Q[new_state])] - Q[state][action_key])
            
            # s <- s'
            state = new_state

            # Reward across the whole episode
            reward_total += reward

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                # Update statistics for the episode
                statistics['rewards'][episode_cnt] = reward_total
                statistics['lap_times'][episode_cnt] = t+1
                if (episode_cnt % snapshot_freq == 0 and episode_cnt != 0):
                    print(f'saving snapshot files to {snapshots_dir}/{episode_cnt:10d}_*.json')
                    save_snapshot(Q,statistics,snapshots_dir,f'{episode_cnt:010d}')
                # break to next episode
                break

    return Q, statistics

def save_snapshot(Q,statistics,directory,filename_prefix):
    # Saves a snapshot to the given directory with the given filename prefix
    # Saves Q in its own file (very large) and statistics in another
    data = {
        'rewards': statistics['rewards'].tolist(),
        'lap_times': statistics['lap_times'].tolist(),
    }
    
    if (os.path.isdir(directory)):
        with open(os.path.join(directory,f'{filename_prefix}_Q.pkl'),'wb') as f:
            pickle.dump(Q, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(directory,f'{filename_prefix}_statistics.json'),'w+') as outfile:
            json.dump(data, outfile)
    else:
        print(f'save_snapshot(): not such file or directory. Snapshot not saved.')
    pass

if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    episodes = 10000
    discount_rate = 0.99
    learning_rate = 0.01
    epsilon = 0.9
    epsilon_decay = .99
    epsilon_floor = .01
    snapshot_freq = 500
    

    now = datetime.datetime.now()
    timestamp = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.microsecond}'

    snapshots_dir = f'snapshots/snapshots_{timestamp}'
    os.mkdir(snapshots_dir)

    # Train the model
    Q, statistics = q_learning_train(
        env,episodes,
        snapshots_dir,
        discount_rate=discount_rate,
        learning_rate=learning_rate,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_floor=epsilon_floor,
        snapshot_freq=snapshot_freq)

    # Save a snapshot of the model and statistics    
    save_snapshot(Q,statistics,snapshots_dir,'FINAL')
    print(f'Final snapshot saved to {snapshots_dir}/FINAL*')

    env.close()