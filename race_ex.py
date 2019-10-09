import gym
import time

env = gym.make('CarRacing-v0')
for i_episode in range(20):
    observation = env.reset()
    start = time.time()
    while(True):
        #env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(f'episode finished in {time.time() - start}')
            break
env.close()
