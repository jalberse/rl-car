import gym
import numpy as np
import cv2 as cv
import sys
from preprocessing import rgb_to_bw_threshold

np.set_printoptions(threshold=sys.maxsize)

env = gym.make('CarRacing-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        thresh = rgb_to_bw_threshold(observation)
        cv.imshow("Observation",thresh)
        cv.waitKey(1)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
