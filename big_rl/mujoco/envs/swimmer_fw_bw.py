import numpy as np
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv as SwimmerEnv_


class SwimmerForwardBackwardEnv(SwimmerEnv_):
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
        xy_position_before = self.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = self._forward_reward_weight * x_velocity * self.target_direction # <-- Modified this line

        ctrl_cost = self.control_cost(action)

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        info = {
            "reward_fwd": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
            "target_direction": self.target_direction, # <-- Added this line
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, False, False, info
