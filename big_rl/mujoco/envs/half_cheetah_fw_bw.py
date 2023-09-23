import numpy as np
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv as HalfCheetahEnv_


class HalfCheetahForwardBackwardEnv(HalfCheetahEnv_):
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

        forward_reward = self._forward_reward_weight * x_velocity * self.target_direction # This is the only line I changed. Everything else is copied from the original environment.

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "target_direction": self.target_direction, # And added this line too, I guess
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info
