import gymnasium
import numpy as np


class MujocoTaskRewardWrapper(gymnasium.Wrapper):
    def __init__(self, env, task_reward_key: str, control_cost_key: str, total_energy: float):
        super().__init__(env)

        self.task_reward_key = task_reward_key
        self.control_cost_key = control_cost_key

        self.total_energy = total_energy

        self._remaining_energy = total_energy

        self.observation_space = gymnasium.spaces.Dict({
            'vector': self.env.observation_space,
            'energy': gymnasium.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        }) # type: ignore

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self._remaining_energy = self.total_energy

        info['energy'] = self._remaining_energy
        return {'vector': obs, 'energy': [self._remaining_energy]}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action) # type: ignore

        reward = info[self.task_reward_key]
        control_cost = info[self.control_cost_key]

        assert control_cost <= 0
        self._remaining_energy += control_cost

        if self._remaining_energy < 0:
            terminated = True
            reward = 0 # Prevents scenarios where the agent tries to expend more energy than it has when it's nearly out of energy.

        info['energy'] = self._remaining_energy
        return {
            'vector': obs,
            'energy': [np.clip(self._remaining_energy, 0, self.total_energy)],
        }, reward, terminated, truncated, info
