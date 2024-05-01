import importlib.resources
from typing import Dict, Tuple, Union

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

with importlib.resources.path("big_rl.mujoco.envs.assets", "ball_1d.xml") as path:
    _xml_file = path.as_posix()


class BallEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        #xml_file: str = "ball_1d.xml",
        xml_file: str = _xml_file,
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,
        #ctrl_cost_weight: float = 0.0,
        #contact_cost_weight: float = 5e-4,
        #healthy_reward: float = 1.0,
        main_body: Union[int, str] = 1,
        #terminate_when_unhealthy: bool = True,
        #healthy_z_range: Tuple[float, float] = (0.2, 1.0),
        #contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        #include_cfrc_ext_in_observation: bool = True,
        exclude_velocity_from_observation=False,
        target_direction : int | None = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            #ctrl_cost_weight,
            #contact_cost_weight,
            #healthy_reward,
            main_body,
            #terminate_when_unhealthy,
            #healthy_z_range,
            #contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            #include_cfrc_ext_in_observation,
            exclude_velocity_from_observation,
            target_direction,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        #self._ctrl_cost_weight = ctrl_cost_weight
        #self._contact_cost_weight = contact_cost_weight

        #self._healthy_reward = healthy_reward
        #self._terminate_when_unhealthy = terminate_when_unhealthy
        #self._healthy_z_range = healthy_z_range

        #self._contact_force_range = contact_force_range

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        #self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation
        self._exclude_velocity_from_observation = exclude_velocity_from_observation

        self._target_direction = target_direction

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # type: ignore (needs to be defined after)
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.qvel.size * (not exclude_velocity_from_observation)

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        if self._target_direction is None:
            self.target_direction = np.random.choice([-1, 1])
        else:
            self.target_direction = self._target_direction

        info["target_direction"] = self.target_direction

        return obs, info

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "target_direction": self.target_direction,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # terminated=False because the ball can never reach an unhealthy state
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = x_velocity * self._forward_reward_weight * self.target_direction
        reward = forward_reward

        reward_info = {
            "reward_forward": forward_reward,
        }

        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._exclude_velocity_from_observation:
            return position
        else:
            return np.concatenate((position, velocity))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }


if __name__ == "__main__":
    env = BallEnv(render_mode='human', exclude_velocity_from_observation=True)
    done = True
    while True:
        if done:
            env.reset()
        _,_,terminated,truncated,_ = env.step(env.action_space.sample())
        #_,_,terminated,truncated,_ = env.step(np.array([1.]))
        done = terminated or truncated
    env.close()
