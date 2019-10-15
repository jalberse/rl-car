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
import time
from collections import defaultdict

np.set_printoptions(threshold=sys.maxsize)

# TODO wrap functions, etc in class so we don't have to rely on a constant like this
ACTION_SPACE_SIZE = 5 # Number of possible actions

def e_greedy(Q,state,epsilon,guided_exploration=False):
    """
    Returns an action key given Q and state
    If guided_exploration is True,
        use a heuristic to dictate left/right/straight
    An action key in {0,..,ACTION_SPACE_SIZE-1}, which is mapped to a possible action in T x G x B
        Returns the optimal action given Q, s with a probability of (1-epsilon)
        Returns a random action with a probability of epsilon
    """
    if (random.random() < 1-epsilon):
        return np.argmax(Q[state]) # Note for ties, will return first index
    else:
        if guided_exploration:
            # Heuristic for exploration:
            #   If any track in the column, set to 1. If no track in column, 0
            #       Choose this over straight column sums because otherwise straight would always dominate
            #   Bucket into 4 4 4 (left, straight, right)
            #   1's in each bucket / total 1's == probability head in direction of that bucket
            state_np = np.asarray(state,dtype=np.uint8)
            expanded = np.unpackbits(state_np)[:-4].reshape(11,12)
            dist = np.sum(expanded,axis=0)
            dist[dist > 0] = 1
            dist = np.sum(dist.reshape(3,4),axis=1)
            dist_total = np.sum(dist)
            left_prob = dist[0] / dist_total
            straight_prob = dist[1] / dist_total / 3 # 3 choices for straight
            right_prob = dist[2] / dist_total 
            r = random.random()
            return np.random.choice(ACTION_SPACE_SIZE,p=[straight_prob,straight_prob,straight_prob,left_prob,right_prob])
        return random.randint(0,ACTION_SPACE_SIZE-1)

def default_action_generator():
    # Replaces typical lambda for defaultdict so we may pickle it
    return np.zeros(ACTION_SPACE_SIZE)


def q_learning_train(env, 
                     num_episodes, 
                     snapshots_dir,
                     discount_rate = 0.99, 
                     learning_rate = 0.01, 
                     epsilon = 0.05, 
                     epsilon_decay = .99, 
                     epsilon_floor = 0, 
                     snapshot_freq=1000,
                     replacing_traces=True,
                     trace_decay=.7,
                     guided_exploration=False):
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
    replacing_traces    Use replacing traces if True (Watkin's Q(lambda)), Q-learning if false
    guided_exploration  If true, use a heuristic to guide exploration when e-greedy results in non-optimal
    """

    # Initialize Q(s,a) for all s in S, a in A (mathematically)
    # In reality we initialize as each state S comes in (many states are unreachable, so don't waste time)
    
    # Action space is [steer, gas, brake] with bounds [[-1,1],[0,1],[0,1]] in real numbers
    # We choose to discretize s.t.
    # steer is in T = {-1, 0, 1} (left, straight, right)
    # gas is in   G = {0, 1} (gas or no gas)
    # and brake   B = {0, 1} (brake or no brake)
    # This discretization should be OK - it is how controls for humans are discretized, after all
    #   Note: I think some literature on this environ also discretizes in this way. Check.
    # Action space is therefore T x G x B less situations where we have both brake and gas active (9 possible options)
    Q = defaultdict(default_action_generator) # So Q[state] is a list of 12 values - a value for each of 12 actions in the state

    # Maps action_key index to action the env can understand
    actions = [
        [0.,0.,0.],  # Coast straight
        [0.,0.,1.],  # Brake straight
        [0.,1.,0.],  # Accelerate straight
        [-1.,0.,0.], # Left turn coast
        [1.,0.,0.],  # Right turn coast
    ]

    # Keep track of information we want to plot later
    statistics = dict([
        ('rewards',np.zeros(num_episodes)), # Total reward obtained this episode
        ('lap_times',np.zeros(num_episodes)), # Timesteps taken in episode
        ('max_reward_in_episode',np.zeros(num_episodes)), # Maximum total reward within each episode (NOT final/total accumulated)
        ('episode_time',np.zeros(num_episodes)),
    ])

    for episode_cnt in range(num_episodes):
        # Reset elibility traces
        if replacing_traces:
            E = defaultdict(lambda: defaultdict(int)) # Eligibility of Q[state][action]
        
        episode_start_time = time.time()
        print(f'Episode: {episode_cnt}')
        if (epsilon > epsilon_floor):
            epsilon = epsilon * epsilon_decay
        print(f'Epsilon: {epsilon}')
        observation = env.reset()
        t = 0
        reward_total = 0
        max_reward = -9999
        # Get the initial state, action
        state = rgb_to_bw_threshold(observation) # State is bitpacked into a uint8 tuple
        action_key = e_greedy(Q,state,epsilon,guided_exploration=guided_exploration)

        render = True

        # Game loop
        while(True):
            t += 1
            if (render):
                env.render()

            # Take action a, observe r, s'
            observation, reward, done, info = env.step(actions[action_key])
            new_state = rgb_to_bw_threshold(observation)

            # Action to take next step
            new_action_key = e_greedy(Q,new_state,epsilon,guided_exploration=guided_exploration)

            if replacing_traces:
                # Watkin's Q with replacing traces
                a_optimal = np.argmax(Q[new_state])
                delta = reward + discount_rate*Q[new_state][a_optimal] - Q[state][action_key]
                E[state][action_key] = 1
                for s in Q:
                    for a in range(len(actions)):
                        Q[s][a] = Q[s][a] + learning_rate*delta*E[s][a]
                        # Trace does not go forever, only until most recent deviation from optimal policy
                        if new_action_key == a_optimal:
                            E[s][a] = discount_rate*trace_decay*E[s][a]
                        else:
                            E[s][a] = 0
            else:
                # Q(s,a) <- Q(s,a) + alpha[r + gamma*max_a'(Q(s',a')) - Q(s,a)]
                Q[state][action_key] = Q[state][action_key] + learning_rate*(reward + discount_rate*Q[new_state][np.argmax(Q[new_state])] - Q[state][action_key])
            
            state = new_state
            action_key = new_action_key

            # Reward across the whole episode
            reward_total += reward

            if reward_total > max_reward:
                max_reward = reward_total

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                # Update statistics for the episode
                print(f'with a reward of {reward_total}')
                statistics['rewards'][episode_cnt] = reward_total
                statistics['lap_times'][episode_cnt] = t+1
                statistics['max_reward_in_episode'][episode_cnt] = max_reward
                elapsed = time.time() - episode_start_time
                statistics['episode_time'][episode_cnt] = elapsed
                
                print(f'Episode finished in {elapsed}')

                if (episode_cnt % snapshot_freq == 0):
                    print(f'saving snapshot files to {snapshots_dir}/{episode_cnt:10d}_*.json')
                    save_snapshot(Q,statistics,snapshots_dir,f'{episode_cnt:010d}')
                # break to next episode
                break

    return Q, statistics

def save_snapshot(Q,statistics,directory,filename_prefix):
    # Saves a snapshot to the given directory with the given filename prefix
    # Saves Q in its own file (very large) and statistics in another

    data = {}
    for key in statistics:
        data[key] = statistics[key].tolist()
    
    if (os.path.isdir(directory)):
        with open(os.path.join(directory,f'{filename_prefix}_Q.pkl'),'wb') as f:
            pickle.dump(Q, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(directory,f'{filename_prefix}_statistics.json'),'w+') as outfile:
            json.dump(data, outfile)
    else:
        print(f'save_snapshot(): not such file or directory. Snapshot not saved.')
    pass

if __name__ == '__main__':
    gym.register(
        id='CarRacing-v1',
        entry_point='car_racing:CarRacing',
        max_episode_steps=1000,
        reward_threshold=900,
    )

    env = gym.make('CarRacing-v1')

    # TODO real argparsing
    episodes = 500000
    discount_rate = 0.99
    learning_rate = 0.01
    epsilon = 0.9
    epsilon_decay = .99
    epsilon_floor = .01
    snapshot_freq = 500
    e_traces = False
    trace_decay = 0.7
    guided = False
    
    if (len(sys.argv) > 1):
        run_name = sys.argv[1]
    else:
        now = datetime.datetime.now()
        run_name = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.microsecond}'
    if (len(sys.argv) > 2):
        if sys.argv[2] == 'E':
            print("Using eligibility traces")
            e_traces = True
    if (len(sys.argv) > 3):
        if sys.argv[3] == 'G':
            print("Using guided exploration")
            guided = True
    

    snapshots_dir = f'snapshots/snapshots_{run_name}'
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
        snapshot_freq=snapshot_freq,
        replacing_traces=e_traces,
        trace_decay=0.7,
        guided_exploration=guided)

    # Save a snapshot of the model and statistics    
    save_snapshot(Q,statistics,snapshots_dir,'FINAL')
    print(f'Final snapshot saved to {snapshots_dir}/FINAL*')

    env.close()