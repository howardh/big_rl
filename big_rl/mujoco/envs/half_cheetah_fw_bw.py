import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.half_cheetah_v4 import DEFAULT_CAMERA_CONFIG
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv as HalfCheetahEnv_


class HalfCheetahOptionalVelocityEnv_v4(HalfCheetahEnv_):
    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        exclude_velocity_from_observation=False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self._exclude_velocity_from_observation = exclude_velocity_from_observation

        obs_dim = 18
        if exclude_current_positions_from_observation:
            obs_dim -= 1
        if exclude_velocity_from_observation:
            obs_dim -= 9
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            "half_cheetah.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )


    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        if self._exclude_velocity_from_observation:
            observation = np.concatenate((position, )).ravel()
        else:
            observation = np.concatenate((position, velocity)).ravel()
        return observation


class HalfCheetahForwardBackwardEnv(HalfCheetahOptionalVelocityEnv_v4):
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
