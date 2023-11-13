import numpy as np
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv as Walker2dEnv_


class Walker2dForwardBackwardEnv(Walker2dEnv_):
    def __init__(self, target_direction=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_direction = target_direction

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        if self._target_direction is None:
            self.target_direction = np.random.choice([-1, 1])
        else:
            self.target_direction = self._target_direction

        info["target_direction"] = self.target_direction

        return obs, info

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity * self.target_direction # <-- Modified this line
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        terminated = self.terminated
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "target_direction": self.target_direction, # <-- Added this line
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info
