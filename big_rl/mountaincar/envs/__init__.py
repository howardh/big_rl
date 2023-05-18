import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

# global variables
NUM_EPISODES = 1000


env = gym.make("MountainCar-v0", render_mode="human")


for _ in range(NUM_EPISODES):
   env.reset()
   while True:
      env.render()
      action = env.action_space.sample()  # agent policy that uses the observation and info
      observation, reward, terminated, truncated, info = env.step(action)
      print(observation, reward, terminated, truncated, info)

      if terminated or truncated:
         break

env.close()

