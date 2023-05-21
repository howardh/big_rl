import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

###
EPISODES = 200
###

# parser
parser = argparse.ArgumentParser()
# parser.add_argument("--env", type=str, default='MountainCar-v0')
parser.add_argument("--test", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--evaulate", type=str, default='./', help='Directory containing the model file')
args = parser.parse_args()


# =====[ TEST ENV ]===== #
if args.test:
   # create mountaincar environment
   env = gym.make("MountainCar-v0", render_mode="human")

   # reset environment
   observation, info = env.reset()

   # run episodes
   for _ in range(200):
      action = env.action_space.sample()  # agent policy that uses the observation and info
      print("Action: ", action)

      observation, reward, terminated, truncated, info = env.step(action)
      print("Observation: ", observation)
      print("Reward: ", reward)

   # termination
   if terminated or truncated:
      print("Terminated")
      observation, info = env.reset()

   env.close()

# =====[ TRAIN ]===== #
if args.train:
   # create mountaincar environment
   env = gym.make("MountainCar-v0")

   # create model
   model = DQN("MlpPolicy", env, verbose=1)

   # train model
   model.learn(total_timesteps=10000)

   # save model
   model.save("big_rl/model/dqn_mountaincar")

   # close environment
   env.close()

# =====[ EVALUATE ]===== #
if args.evaluate:
   # create mountaincar environment
   env = gym.make("MountainCar-v0")

   # load model
   model = DQN.load(args.model_dir + "dqn_mountaincar")

   # evaluate model
   mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

   # print results
   print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

   # close environment
   env.close()