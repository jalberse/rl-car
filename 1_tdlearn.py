import gym
from tqdm import tqdm
import numpy as np
from preprocessing import rgb_to_bw_threshold
from collections import defaultdict

def generate_sars():
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    next_state = rgb_to_bw_threshold(observation)
    return action, reward, next_state, done


def tdlearn_train(
        env,
        episodes = 2, 
        timesteps = 1000,
        discount_factor = 0.90, 
        learning_rate = 0.01
        ):

    V = defaultdict(float)
    deltas = defaultdict(list)
    A = defaultdict(list)

    for i_episode in tqdm(range(episodes)):
        
            print("Episode {} begins:".format(i_episode))
            
            # Episode Initiation
            t = 0
            rewards = 0.0
            observation = env.reset()
            state = rgb_to_bw_threshold(observation) 

            # Policy Evaluation
            while(True):
                action, reward, next_state, done = generate_sars()
                A[state].append(action)
                rewards += reward
                t += 1
                if done:
                    print("Episode finished after {} timesteps".format(t))
                    print("Total Reward:",rewards)
                    break

                # Modify Value function
                V_old = V[state]
                V[state] = V[state] + learning_rate*(reward + discount_factor*V[next_state] - V[state])
                deltas[state].append(float(np.abs(V_old - V[state])))
                state = next_state

            # Policy Improvement
            #policy_stable = true
            #for state in V.keys():
            #    old_action = V

    return V, A, deltas

    
if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    episodes = 2
    timesteps = 1000
    discount_factor = 0.90
    learning_rate = 0.01
    
    
V, A, deltas  = tdlearn_train(
        env,
        episodes,
        timesteps,
        discount_factor = discount_factor,
        learning_rate = learning_rate
        )



print("VFunc: Num of states updated: ",len(V.keys()))
print("VFunc: Num of Values updated: ", len(V.values()))
print("Delta: Num of states updated:",len(deltas.keys()))
print("Delta: Num of values updated: ", len(deltas.values()))
print("Action: Num of states updated:",len(A.keys()))
print("Action: Num of values updated: ", len(A.values()))




env.close()
