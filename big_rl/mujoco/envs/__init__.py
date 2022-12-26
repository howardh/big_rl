from typing import List

from bs4 import BeautifulSoup
import cv2
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict
import numpy as np
import tempfile


def make_env(env_name: str,
        config={},
        #mujoco_config={},
        meta_config=None) -> gym.Env:
    env = gym.make(env_name, render_mode='rgb_array', **config)
    if meta_config is not None:
        #env = MetaWrapper(env, **meta_config)
        raise NotImplementedError()
    return env


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class MjcfModelFactory():
    def __init__(self, model_name):
        self.grid_size = (10, 10)
        self.cell_size = 2
        self.height = 3

        self.soup = BeautifulSoup(features='xml')

        self.root = self._create_root(model_name)
        self.default = self._create_default()
        self.asset = self._create_asset()
        self.worldbody = self.soup.new_tag('worldbody')
        self.actuator = self.soup.new_tag('actuator')

        self.root.append(self.default)
        self.root.append(self.asset)
        self.root.append(self.worldbody)
        self.root.append(self.actuator)

        self.agent = None # Body that can be controlled
        self.walls = []
        self.balls = []
        self.boxes = []

    def to_xml(self):
        return self.soup.prettify()

    def _create_root(self, model_name):
        root = self.soup.new_tag('mujoco', attrs={'model': model_name})

        root.append(self.soup.new_tag('compiler', attrs={
            'angle': 'degree',
            'coordinate': 'local',
            'inertiafromgeom': 'true',
        }))
        root.append(self.soup.new_tag('option', attrs={
            'integrator': 'RK4',
            'timestep': '0.01'
        }))

        self.soup.append(root)

        return root

    def _create_default(self):
        default = self.soup.new_tag('default')
        default.append(self.soup.new_tag('joint', attrs={
            'armature': '1',
            'damping': '1',
            'limited': 'true'
        }))
        default.append(self.soup.new_tag('geom', attrs={
            'conaffinity': '0',
            'condim': '3',
            'density': '5',
            'friction': '1 0.5 0.5',
            'margin': '0.01',
            'rgba': '0.8 0.6 0.4 1',
        }))

        return default

    def _create_asset(self):
        asset = self.soup.new_tag('asset')
        asset.append(self.soup.new_tag('texture', attrs={
            'builtin': 'gradient',
            'height': '100',
            'rgb1': '1 1 1',
            'rgb2': '0 0 0',
            'type': 'skybox',
            'width': '100'
        }))
        asset.append(self.soup.new_tag('texture', attrs={
            'builtin': 'flat',
            'height': '1278',
            'mark': 'cross',
            'markrgb': '1 1 1',
            'name': 'texgeom',
            'random': '0.01',
            'rgb1': '0.8 0.6 0.4',
            'rgb2': '0.8 0.6 0.4',
            'type': 'cube',
            'width': '127'
        }))
        asset.append(self.soup.new_tag('texture', attrs={
            'builtin': 'checker',
            'height': '100',
            'name': 'texplane',
            'rgb1': '0 0 0',
            'rgb2': '0.8 0.8 0.8',
            'type': '2d',
            'width': '100'
        }))
        asset.append(self.soup.new_tag('material', attrs={
            'name': 'MatPlane',
            'reflectance': '0.5',
            'shininess': '1',
            'specular': '1',
            'texrepeat': '60 60',
            'texture': 'texplane'
        }))
        asset.append(self.soup.new_tag('material', attrs={
            'name': 'geom',
            'texture': 'texgeom',
            'texuniform': 'true'
        }))

        return asset

    def add_ground(self):
        self.worldbody.append(self.soup.new_tag('geom', attrs={
            'conaffinity': '1',
            'condim': '3',
            'material': 'MatPlane',
            'name': 'floor',
            'pos': '0 0 0',
            'rgba': '0.8 0.9 0.8 1',
            'size': '40 40 40',
            'type': 'plane',
        }))

    def add_ceiling(self):
        self.worldbody.append(self.soup.new_tag('geom', attrs={
            'conaffinity': '1',
            'condim': '3',
            'material': 'MatPlane',
            'name': 'floor',
            'pos': f'0 0 {self.height}',
            'rgba': '0 0 0 0',
            'size': '40 40 40',
            'type': 'plane',
        }))

    def add_light(self):
        self.worldbody.append(self.soup.new_tag('light', attrs={
            'cutoff': '100',
            'diffuse': '1 1 1',
            'dir': '0 0 -1',
            'exponent': '1',
            'pos': '0 0 3',
            'specular': '1 1 1'
        }))

    def add_ant(self, pos, name_prefix='', show_nose = False, num_legs=4):
        body_name = f'{name_prefix}torso'
        body = self.soup.new_tag('body', attrs={
            'name': body_name,
            'pos': f'{pos[0] * self.cell_size} {pos[1] * self.cell_size} 0.75'
        })
        actuators = []

        camera_track = self.soup.new_tag(
            'camera',
            attrs={
                'name': name_prefix+'track',
                'mode': 'trackcom',
                'pos': '0 -3 0.3',
                'xyaxes': '1 0 0 0 0 1'
            }
        )
        body.append(camera_track)

        camera_first_person = self.soup.new_tag('camera', attrs={'name': name_prefix+'first_person', 'mode': 'fixed', 'pos': '0 0 0', 'axisangle': '1 0 0 90'})
        body.append(camera_first_person)

        torso_geom = self.soup.new_tag('geom', attrs={'name': name_prefix+'torso_geom', 'pos': '0 0 0', 'size': '0.25', 'type': 'sphere'})
        body.append(torso_geom)

        # Make it a free joint so that we can move it around
        root_joint = self.soup.new_tag('joint', attrs={'armature': '0', 'damping': '0', 'limited': 'false', 'margin': '0.01', 'name': name_prefix+'root_joint', 'pos': '0 0 0', 'type': 'free'})
        body.append(root_joint)

        # Nose (for debugging purposes. Shows the direction of the first person camera.)
        if show_nose:
            body.append(self.soup.new_tag('geom', attrs={
                'contype': '0',
                'conaffinity': '0',
                'name': name_prefix+'nose_geom',
                'fromto': '0 0 0.1 0 10 0',
                'size': '0.01',
                'type': 'capsule',
                'rgba': '1 0 0 0.5',
                'mass': '0',
            }))

        # Legs
        angles = np.linspace(0, 360, num_legs, endpoint=False)
        for i in range(num_legs):
            leg = self.soup.new_tag('body', attrs={
                'name': f'{name_prefix}leg{i}',
                'pos': '0 0 0',
                'axisangle': f'0 0 1 {angles[i]}'
            })
            body.append(leg)

            leg_aux_geom = self.soup.new_tag('geom', attrs={
                'fromto': '0.0 0.0 0.0 0.2 0.2 0.0',
                'name': f'{name_prefix}leg{i}_aux_geom',
                'size': '0.08',
                'type': 'capsule'
            })
            leg.append(leg_aux_geom)

            leg_aux = self.soup.new_tag('body', attrs={
                'name': f'{name_prefix}leg{i}_aux1',
                'pos': '0.2 0.2 0'
            })
            leg.append(leg_aux)

            leg_hip = self.soup.new_tag('joint', attrs={
                'axis': f'0 0 1',
                'name': f'{name_prefix}hip{i}',
                'pos': '0.0 0.0 0.0',
                'range': '-30 30',
                'type': 'hinge'
            })
            leg_aux.append(leg_hip)
            actuators.append(self.soup.new_tag('motor', attrs={
                'ctrllimited': 'true',
                'ctrlrange': '-1 1',
                'joint': f'{name_prefix}hip{i}',
                'gear': '150',
            }))

            leg_geom = self.soup.new_tag('geom', attrs={
                'fromto': '0.0 0.0 0.0 0.2 0.2 0.0',
                'name': f'{name_prefix}leg{i}_geom',
                'size': '0.08',
                'type': 'capsule'
            })
            leg_aux.append(leg_geom)

            leg_body = self.soup.new_tag('body', attrs={'pos': '0.2 0.2 0', 'name': f'{name_prefix}leg{i}_aux2'})
            leg_aux.append(leg_body)

            leg_ankle = self.soup.new_tag('joint', attrs={
                'axis': '-1 1 0',
                'name': f'{name_prefix}ankle{i}',
                'pos': '0.0 0.0 0.0',
                'range': '30 70',
                'type': 'hinge'
            })
            leg_body.append(leg_ankle)
            actuators.append(self.soup.new_tag('motor', attrs={
                'ctrllimited': 'true',
                'ctrlrange': '-1 1',
                'joint': f'{name_prefix}ankle{i}',
                'gear': '150',
            }))

            leg_ankle_geom = self.soup.new_tag('geom', attrs={
                'fromto': '0.0 0.0 0.0 0.4 0.4 0.0',
                'name': f'{name_prefix}ankle{i}_geom',
                'size': '0.08',
                'type': 'capsule'
            })
            leg_body.append(leg_ankle_geom)

        self.worldbody.append(body)
        for a in actuators:
            self.actuator.append(a)
        self.agent = { 'body': body, 'actuators': actuators }

        return body_name

    def add_wall(self, cell, cell2=None):
        if cell2 is None:
            pos = (
                cell[0] * self.cell_size,
                cell[1] * self.cell_size,
                self.height / 2,
            )
            size = (self.cell_size / 2, self.cell_size / 2, self.height / 2)
        else:
            pos = (
                (cell[0] + cell2[0]) * self.cell_size / 2,
                (cell[1] + cell2[1]) * self.cell_size / 2,
                self.height / 2,
            )
            size = (
                (abs(cell[0] - cell2[0]) + 1) * self.cell_size / 2,
                (abs(cell[1] - cell2[1]) + 1) * self.cell_size / 2,
                self.height / 2,
            )
        #print(cell, cell2, pos, size)
        #self.worldbody.append(self.soup.new_tag('geom', attrs={
        #    'conaffinity': '0',
        #    'condim': '3',
        #    'size': '0.08',
        #    'fromto': f'{pos[0]} {pos[1]} 0 {pos[0]} {pos[1]} 5',
        #    'type': 'capsule',
        #    'rgba': '0 1 0 1',
        #}))
        wall = self.soup.new_tag('geom', attrs={
            'conaffinity': '1',
            'condim': '3',
            'material': 'MatPlane',
            'name': f'wall_{len(self.walls)}',
            'pos': ' '.join([str(p) for p in pos]),
            'size': ' '.join([str(s) for s in size]),
            'type': 'box',
            'rgba': '0.8 0.8 0.8 0.5',
        })
        self.walls.append(wall)
        self.worldbody.append(wall)

    def add_ball(self, pos, size=0.5, rgba=[1, 0, 0, 1]):
        idx = len(self.balls)
        size = 0.5
        body_name = f'ball{idx}'

        coords = [pos[0] * self.cell_size, pos[1] * self.cell_size, size]
        body = self.soup.new_tag('body', attrs={'name': body_name, 'pos': ' '.join(map(str, coords))})

        body.append(self.soup.new_tag('joint', attrs={
            'armature': '0',
            'damping': '0',
            'limited': 'false',
            'margin': '0.01',
            'name': f'ball{idx}_joint',
            'pos': '0 0 0',
            'type': 'free'
        }))
        body.append(self.soup.new_tag('geom', attrs={
            'conaffinity': '1',
            'name': f'ball{idx}_geom',
            'pos': '0 0 0',
            'size': str(size),
            'type': 'sphere',
            'rgba': ' '.join(map(str, rgba))
        }))

        self.balls.append(body)
        self.worldbody.append(body)

        return body_name

    def add_box(self, pos, size=0.5, rgba=[1, 0, 0, 1]):
        idx = len(self.boxes)
        size = 0.5
        body_name = f'box{idx}'

        coords = [pos[0] * self.cell_size, pos[1] * self.cell_size, size]
        body = self.soup.new_tag('body', attrs={'name': body_name, 'pos': ' '.join(map(str, coords))})

        body.append(self.soup.new_tag('joint', attrs={
            'armature': '0',
            'damping': '0',
            'limited': 'false',
            'margin': '0.01',
            'name': f'box{idx}_joint',
            'pos': '0 0 0',
            'type': 'free'
        }))
        body.append(self.soup.new_tag('geom', attrs={
            'conaffinity': '1',
            'name': f'box{idx}_geom',
            'pos': '0 0 0',
            'size': f'{size} {size} {size}',
            'type': 'box',
            'rgba': ' '.join(map(str, rgba))
        }))

        self.balls.append(body)
        self.worldbody.append(body)

        return body_name


class AntBaselineEnv(MujocoEnv, utils.EzPickle):
    """ Copied from https://github.com/Farama-Foundation/Gymnasium/blob/d71a13588266256a4c900b5e0d72d10785816c3a/gymnasium/envs/mujoco/ant_v4.py """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        use_internal_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        camera = 'first_person',
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            ctrl_cost_weight,
            use_contact_forces,
            use_internal_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            camera,
            **kwargs
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces
        self._use_internal_forces = use_internal_forces

        self._camera = camera
        self._image_size = 56

        # Init MJCF file
        mjcf = MjcfModelFactory('ant')
        mjcf.add_ground()
        mjcf.add_light()
        self._agent_body_name = mjcf.add_ant([0,0], num_legs=4)

        with tempfile.NamedTemporaryFile(suffix='.xml') as f:
            f.write(mjcf.to_xml().encode('utf-8'))
            f.flush()

            MujocoEnv.__init__(
                self,
                model_path=f.name,
                frame_skip=5,
                observation_space=None,
                default_camera_config=DEFAULT_CAMERA_CONFIG,
                **kwargs
            )

        self._compute_indices([self._agent_body_name])

        sample_obs = self._get_obs()
        observation_space_dict = {
            'state': Box(
                low=-np.inf, high=np.inf, shape=sample_obs['state'].shape, dtype=np.float64
            ),
        }
        if 'image' in sample_obs:
            observation_space_dict['image'] = Box(
                low=0, high=255, shape=sample_obs['image'].shape, dtype=np.uint8
            )
        self.observation_space = Dict(observation_space_dict)


    def _compute_indices(self, bodies):
        """
        Compute the indices of the qpos, qvel, cfrc_int, and cfrc_ext arrays that correspond to the bodies specified in the constructor. The resulting indices are saved in `self._qpos_adr` and `self._qvel_adr`. They can be used to get the relevant qpos and qvel with `qpos = env.data.qpos[self._qpos_adr]` and `qvel = env.data.qvel[self._qvel_adr]`
        """
        env = self

        # Get all joints associated with the bodies named in `bodies` and all child bodies
        joint_ids = set()
        body_ids = set()
        for body_name in bodies:
            joint_ids.update(list_joints_with_ancestor(env, env.model.body(body_name).id))
            body_ids.update(list_bodies_with_ancestor(env, env.model.body(body_name).id))

        # Get qpos and qvel indices for all joints in `joint_ids`
        qpos_adr: List[int] = []
        qvel_adr: List[int] = []
        for joint_id in joint_ids:
            # qpos
            start_adr = env.model.jnt_qposadr[joint_id]
            try:
                # Get next joint address
                end_adr = min(adr for adr in env.model.jnt_qposadr if adr > start_adr)
            except ValueError:
                # Raises a ValueError if `min` receives an empty sequence
                end_adr = env.model.nq # dim(qpos)
            for adr in range(start_adr, end_adr):
                qpos_adr.append(adr)
            # qvel
            start_adr = env.model.jnt_dofadr[joint_id]
            try:
                # Get next joint address
                end_adr = min(adr for adr in env.model.jnt_dofadr if adr > start_adr)
            except ValueError:
                # Raises a ValueError if `min` receives an empty sequence
                end_adr = env.model.nv
            for adr in range(start_adr, end_adr):
                qvel_adr.append(adr)

        # Save the indices
        self._qpos_adr = np.array(list(sorted(qpos_adr)))
        self._qvel_adr = np.array(list(sorted(qvel_adr)))
        self._body_ids = np.array(list(sorted(body_ids)))

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext[self._body_ids,:].flatten()
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.get_body_com(self._agent_body_name)
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, action):
        xy_position_before = self.get_body_com(self._agent_body_name)[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com(self._agent_body_name)[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs_state_dict(self):
        position = self.data.qpos[self._qpos_adr]
        velocity = self.data.qvel[self._qvel_adr]
        contact_force = self.contact_forces
        internal_force = self.data.cfrc_int[self._body_ids,:].flatten()
        obs = {
            "position": position,
            "velocity": velocity,
            "contact_force": contact_force,
            "internal_force": internal_force,
        }

        if self._camera is not None:
            rgb_array = self.mujoco_renderer.render(
                    render_mode='rgb_array',
                    camera_name=self._camera
            )

            # Resize
            if self._image_size is not None:
                obs['image'] = cv2.resize( # type: ignore
                    rgb_array,
                    (self._image_size, self._image_size),
                    interpolation=cv2.INTER_AREA, # type: ignore
                )

            # Move channel dimension to start
            obs['image'] = np.moveaxis(obs['image'], 2, 0)

        return obs

    def _get_obs(self):
        obs_dict = self._get_obs_state_dict()
        obs = {}

        state = [obs_dict['position'], obs_dict['velocity']]
        if self._use_contact_forces:
            state.append(obs_dict['contact_force'])
        if self._use_internal_forces:
            state.append(obs_dict['internal_force'])
        obs['state'] = np.concatenate(state)

        if self._camera is not None:
            obs['image'] = obs_dict['image']

        return obs

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


class AntFetchEnv(MujocoEnv, utils.EzPickle):
    """ Copied from https://github.com/Farama-Foundation/Gymnasium/blob/d71a13588266256a4c900b5e0d72d10785816c3a/gymnasium/envs/mujoco/ant_v4.py """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        use_internal_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 2.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        camera = 'first_person',
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            ctrl_cost_weight,
            use_contact_forces,
            use_internal_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            camera,
            **kwargs
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces
        self._use_internal_forces = use_internal_forces

        self._camera = camera
        self._image_size = 56

        # Init MJCF file
        mjcf = MjcfModelFactory('ant')
        mjcf.add_ground()
        mjcf.add_light()
        mjcf.add_wall([0,0], [5,0])
        mjcf.add_wall([0,0], [0,5])
        mjcf.add_wall([5,5], [0,5])
        mjcf.add_wall([5,5], [5,0])
        self._agent_body_name = mjcf.add_ant([1,1], num_legs=4)
        self._objects = [
            mjcf.add_ball([3,3]),
            mjcf.add_ball([2,3]),
            mjcf.add_ball([2,1]),
            mjcf.add_box([1,2]),
        ]

        with tempfile.NamedTemporaryFile(suffix='.xml') as f:
            f.write(mjcf.to_xml().encode('utf-8'))
            f.flush()

            MujocoEnv.__init__(
                self,
                model_path=f.name,
                frame_skip=5,
                observation_space=None,
                default_camera_config=DEFAULT_CAMERA_CONFIG,
                **kwargs
            )

        self._compute_indices([self._agent_body_name])

        sample_obs = self._get_obs()
        observation_space_dict = {
            'state': Box(
                low=-np.inf, high=np.inf, shape=sample_obs['state'].shape, dtype=np.float64
            ),
        }
        if 'image' in sample_obs:
            observation_space_dict['image'] = Box(
                low=0, high=255, shape=sample_obs['image'].shape, dtype=np.uint8
            )
        self.observation_space = Dict(observation_space_dict)


    def _compute_indices(self, bodies):
        """
        Compute the indices of the qpos, qvel, cfrc_int, and cfrc_ext arrays that correspond to the bodies specified in the constructor. The resulting indices are saved in `self._qpos_adr` and `self._qvel_adr`. They can be used to get the relevant qpos and qvel with `qpos = env.data.qpos[self._qpos_adr]` and `qvel = env.data.qvel[self._qvel_adr]`
        """
        env = self

        # Get all joints associated with the bodies named in `bodies` and all child bodies
        joint_ids = set()
        body_ids = set()
        for body_name in bodies:
            joint_ids.update(list_joints_with_ancestor(env, env.model.body(body_name).id))
            body_ids.update(list_bodies_with_ancestor(env, env.model.body(body_name).id))

        # Get qpos and qvel indices for all joints in `joint_ids`
        qpos_adr: List[int] = []
        qvel_adr: List[int] = []
        for joint_id in joint_ids:
            # qpos
            start_adr = env.model.jnt_qposadr[joint_id]
            try:
                # Get next joint address
                end_adr = min(adr for adr in env.model.jnt_qposadr if adr > start_adr)
            except ValueError:
                # Raises a ValueError if `min` receives an empty sequence
                end_adr = env.model.nq # dim(qpos)
            for adr in range(start_adr, end_adr):
                qpos_adr.append(adr)
            # qvel
            start_adr = env.model.jnt_dofadr[joint_id]
            try:
                # Get next joint address
                end_adr = min(adr for adr in env.model.jnt_dofadr if adr > start_adr)
            except ValueError:
                # Raises a ValueError if `min` receives an empty sequence
                end_adr = env.model.nv
            for adr in range(start_adr, end_adr):
                qvel_adr.append(adr)

        # Save the indices
        self._qpos_adr = np.array(list(sorted(qpos_adr)))
        self._qvel_adr = np.array(list(sorted(qvel_adr)))
        self._body_ids = np.array(list(sorted(body_ids)))

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext[self._body_ids,:].flatten()
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.get_body_com(self._agent_body_name)
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, action):
        xy_position_before = self.get_body_com(self._agent_body_name)[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com(self._agent_body_name)[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs_state_dict(self):
        position = self.data.qpos[self._qpos_adr]
        velocity = self.data.qvel[self._qvel_adr]
        contact_force = self.contact_forces
        internal_force = self.data.cfrc_int[self._body_ids,:].flatten()
        obs = {
            "position": position,
            "velocity": velocity,
            "contact_force": contact_force,
            "internal_force": internal_force,
        }

        if self._camera is not None:
            rgb_array = self.mujoco_renderer.render(
                    render_mode='rgb_array',
                    camera_name=self._camera
            )

            # Resize
            if self._image_size is not None:
                obs['image'] = cv2.resize( # type: ignore
                    rgb_array,
                    (self._image_size, self._image_size),
                    interpolation=cv2.INTER_AREA, # type: ignore
                )

            # Move channel dimension to start
            obs['image'] = np.moveaxis(obs['image'], 2, 0)

        return obs

    def _get_obs(self):
        obs_dict = self._get_obs_state_dict()
        obs = {}

        state = [obs_dict['position'], obs_dict['velocity']]
        if self._use_contact_forces:
            state.append(obs_dict['contact_force'])
        if self._use_internal_forces:
            state.append(obs_dict['internal_force'])
        obs['state'] = np.concatenate(state)

        if self._camera is not None:
            obs['image'] = obs_dict['image']

        return obs

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

    def get_collisions(self):
        """  """
        torso_id = self.model.body(self._agent_body_name).id
        collided_objects = []

        for g1,g2 in zip(self.data.contact.geom1,self.data.contact.geom2):
            body1 = self.model.body(self.model.geom_bodyid[g1])
            body2 = self.model.body(self.model.geom_bodyid[g2])

            body1_root = body1.rootid.item()
            body2_root = body2.rootid.item()

            if body1_root == 0:
                continue
            if body2_root == 0:
                continue

            if body1_root == torso_id:
                collided_objects.append(body2.name)
            elif body2_root == torso_id:
                collided_objects.append(body1.name)

        return collided_objects

    def get_sq_dist_to_objects(self, pos):
        dist = {}
        for obj_body_name in self._object_names:
            obj_pos = self.get_body_com(obj_body_name)
            sq_dist = np.sum(np.square(pos - obj_pos))
            dist[obj_body_name] = sq_dist
        return dist

    def place_object(self, object_name, position=None, orientation=None):
        """  """
        if position is None:
            position = np.random.uniform(low=1, high=4, size=3)
            position[2] = 3
        if orientation is None:
            # Generate a random quaternion
            orientation = np.random.uniform(low=-1, high=1, size=4)
        # TODO: Look for the the joint associated with the body instead of guessing at the name
        adr = self.model.joint(f'{object_name}_joint').qposadr.item()
        self.data.qpos[adr:adr+3] = position
        self.data.qpos[adr+3:adr+7] = orientation


register(
    id="AntBaselineEnv-v0",
    entry_point="big_rl.mujoco.envs:AntBaselineEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="AntFetchEnv-v0",
    entry_point="big_rl.mujoco.envs:AntFetchEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)


def strz(x):
    """ Truncate a null-terminated string. """
    for i,c in enumerate(x):
        if c == 0:
            return x[:i]
    return x


def list_geoms_by_body(env, body_id=0, depth=0, covered_bodies=None):
    if covered_bodies is None:
        covered_bodies = set()
    if body_id in covered_bodies:
        return
    covered_bodies.add(body_id)

    body_name_adr = env.model.name_bodyadr[body_id]
    body_name = strz(env.model.names[body_name_adr:])
    print(f'{"  "*depth} {body_id} {body_name}')
    for geom_id,geom_name_adr in enumerate(env.model.name_geomadr):
        geom_name = strz(env.model.names[geom_name_adr:])
        if env.model.geom_bodyid[geom_id] == body_id:
            print(f'{"  "*depth}   {geom_id} {geom_name}')
    for child_id,parent_id in enumerate(env.model.body_parentid):
        if parent_id == body_id:
            list_geoms_by_body(child_id, depth+1, covered_bodies)


def list_joints_with_ancestor(env, body_id, depth=0, covered_bodies=None):
    joint_ids = set()

    if covered_bodies is None:
        covered_bodies = set()
    if body_id in covered_bodies:
        return joint_ids
    covered_bodies.add(body_id)

    body_name_adr = env.model.name_bodyadr[body_id]
    body_name = strz(env.model.names[body_name_adr:])
    print(f'{"  "*depth} {body_id} {body_name}')
    for joint_id,joint_name_adr in enumerate(env.model.name_jntadr):
        joint_name = strz(env.model.names[joint_name_adr:])
        if env.model.jnt_bodyid[joint_id] == body_id:
            print(f'{"  "*depth}   {joint_id} {joint_name}')
            joint_ids.add(joint_id)
    for child_id,parent_id in enumerate(env.model.body_parentid):
        if parent_id == body_id:
            j = list_joints_with_ancestor(env, child_id, depth+1, covered_bodies)
            joint_ids.update(j)
    return joint_ids


def list_bodies_with_ancestor(env, body_id, depth=0):
    body_ids = set([body_id])

    body_name_adr = env.model.name_bodyadr[body_id]
    body_name = strz(env.model.names[body_name_adr:])
    print(f'{"  "*depth} {body_id} {body_name}')
    for child_id,parent_id in enumerate(env.model.body_parentid):
        if parent_id == body_id:
            body_ids.update(list_bodies_with_ancestor(env, child_id, depth+1))
    return body_ids


if __name__ == '__main__':
    #env = gym.make('AntFetchEnv-v0', render_mode='human')
    env = gym.make('AntBaselineEnv-v0', render_mode='human')
    env.reset()
    print(env.action_space)
    total_reward = 0
    for i in range(1000):
        env.render()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        total_reward += reward
        done = terminated or truncated
        if done:
            print(f'Episode terminated with reward {total_reward}')
            total_reward = 0
            obs, info = env.reset()

        #collided_objects = env.get_collisions()
        #if len(collided_objects) > 0:
        #    print('Collisions:', collided_objects)
        #for obj in collided_objects:
        #    env.place_object(obj)


        ## Display image
        #import matplotlib
        #from matplotlib import pyplot as plt
        #plt.imshow(obs['image'].transpose(1,2,0))
        #plt.show()

        ## Save image
        #import imageio
        #imageio.imwrite(f'frame_{i:04d}.png', rgb_array)

        #breakpoint()
