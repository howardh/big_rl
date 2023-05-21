import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy



# =====[ PARSER ]===== #
parser = argparse.ArgumentParser()
# parser.add_argument("--env", type=str, default='MountainCar-v0')
parser.add_argument("--run", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--evaluate", type=str, help='Directory containing the model file')
parser.add_argument("--model_type", type=str, default='.zip')
# parser.add_argument("--render", action="store_true")
args = parser.parse_args()


# =====[ RUN ENV ]===== #
if args.run:
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

   if args.model_type == '.zip':
      # load model
      model = DQN.load(args.evaluate)

      # evaluate model
      mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

      # print results
      print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

   elif args.model_type == '.pt':

      # load model
      model = torch.load(args.evaluate, map_location=torch.device('cpu'))

      # evaluate model
      rewards = []
      for _ in range(10):
         obs = env.reset()
         episode_reward = 0
         while True:
            action = model.select_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                  break
         rewards.append(episode_reward)

      mean_reward = np.mean(rewards)
      std_reward = np.std(rewards)

      # print results
      print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

   # close environment
   env.close()