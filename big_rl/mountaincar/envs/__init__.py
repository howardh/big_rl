import gym 
env = gym.make('MountainCar-v0')

observation = env.reset()
print(observation)
for t in range(5):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated , info = env.step(action)
    print(next_state, reward, terminated, truncated , info)
env.close()