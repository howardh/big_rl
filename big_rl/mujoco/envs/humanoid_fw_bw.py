import numpy as np
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv as HumanoidEnv_
from gymnasium.envs.mujoco.humanoid_v4 import mass_center


class HumanoidForwardBackwardEnv(HumanoidEnv_):
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
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity * self.target_direction # <-- Changed this line
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        observation = self._get_obs()
        reward = rewards - ctrl_cost
        terminated = self.terminated
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info
