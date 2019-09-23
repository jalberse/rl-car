import gym
import numpy as np
import cv2 as cv
import sys
from preprocessing import rgb_to_bw_threshold

np.set_printoptions(threshold=sys.maxsize)

env = gym.make('CarRacing-v0')
for i_episode in range(20):
    observation = env.reset()
    t = 0
    while(True):
        t += 1
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        state = rgb_to_bw_threshold(observation)
        # TODO: Take this out when we train I think it might be slowing us down
        cv.imshow("Observation",state)
        cv.waitKey(1)
        # TODO: Terminating after 1000 timesteps rather than just when done or exiting playfield. Why?
        #       Our local version of car_racing.py might differ from what I'm actually importing. 
        #       Check out that code and how it's setting done
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
