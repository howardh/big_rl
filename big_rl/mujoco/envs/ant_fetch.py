from collections import defaultdict
from typing import List

import cv2
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict
import numpy as np
import tempfile
import mujoco

from big_rl.mujoco.envs import MjcfModelFactory, list_joints_with_ancestor, list_bodies_with_ancestor, is_upsidedown


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
        healthy_if_flipped=True,
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        camera = 'first_person',
        num_objs = 2,
        num_target_objs = 1,
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
            healthy_if_flipped,
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
        self._healthy_if_flipped = healthy_if_flipped

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces
        self._use_internal_forces = use_internal_forces

        self._camera = camera
        self._image_size = 56

        self.num_objs = num_objs
        self.num_target_objs = num_target_objs

        # Init MJCF file
        mjcf = MjcfModelFactory('ant')
        mjcf.add_ground()
        mjcf.add_light(pos=[2.5,2.5,5])
        mjcf.add_wall([0,0], [5,0])
        mjcf.add_wall([0,5], [5,5])
        mjcf.add_wall([0,1], [0,4])
        mjcf.add_wall([5,1], [5,4])
        mjcf.set_boundary(x_min=0, x_max=5, y_min=0, y_max=5)
        self._agent_body_name = mjcf.add_ant([2.5,2.5], num_legs=4)
        self._objects = []
        self._object_desc = {}
        colours = {
                'red': (1,0,0,1),
                'green': (0,1,0,1),
                'blue': (0,0,1,1),
        }
        for colour_name, rgba in colours.items():
            obj_name = mjcf.add_box([0,0], rgba=rgba)
            self._objects.append(obj_name)
            self._object_desc[obj_name] = f'{colour_name} box'

            obj_name = mjcf.add_ball([0,0], rgba=rgba)
            self._objects.append(obj_name)
            self._object_desc[obj_name] = f'{colour_name} ball'
        self._cell_size = mjcf.cell_size

        with tempfile.NamedTemporaryFile(suffix='.xml') as f:
            f.write(mjcf.to_xml().encode('utf-8'))
            f.flush()

            MujocoEnv.__init__(
                self,
                model_path=f.name,
                frame_skip=5,
                observation_space=None,
                default_camera_config={
                    'distance': 20.0,
                    #'lookat': [5, 5, 0], # TODO: Make this point to centre
                    'elevation': -45.0,
                    "trackbodyid": 1,
                    'type': mujoco.mjtCamera.mjCAMERA_TRACKING,
                },
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

        # Check rotation
        if not self._healthy_if_flipped:
            body_id = self.model.body(self._agent_body_name).id
            body_jnt = self.model.jnt_bodyid == body_id
            qpos_adr = self.model.jnt_qposadr[body_jnt].item()
            quat = self.data.qpos[qpos_adr + 3 : qpos_adr + 7]
            is_healthy = is_healthy and not is_upsidedown(quat, threshold=-0.9)

        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        # Choose a number of objects to place to be accessible by the agent and hide the rest outside.
        for i,obj in enumerate(self._objects):
            self.place_object(obj, position=[i,-2], height=1)
        self.target_objects = []
        for i in np.random.choice(len(self._objects), size=self.num_objs, replace=False):
            self.place_object(self._objects[i])
            if len(self.target_objects) < self.num_target_objs:
                self.target_objects.append(self._objects[i])
        target_obj_desc = [self._object_desc[obj] for obj in self.target_objects]
        self.goal_str = f'Fetch the {" or ".join(target_obj_desc)}'

        # Stats
        self._fetch_reward_total = 0
        self._objects_picked_up = {k:0 for k in self._objects}

        return obs, info

    def step(self, action):
        # Standard Mujoco rewards
        xy_position_before = self.get_body_com(self._agent_body_name)[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com(self._agent_body_name)[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        # Task-specific
        # If the agent touches an object, it gets a reward and the object is dropped in a new random location.
        collided_objects = self.get_collisions()
        fetch_reward = 0
        for obj in collided_objects:
            self.place_object(obj)
            if obj in self.target_objects:
                fetch_reward += 1
            else:
                fetch_reward -= 1
            self._fetch_reward_total += fetch_reward
            self._objects_picked_up[obj] += 1

        # Build return values
        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "reward_fetch": fetch_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,

            "objects_picked_up": self._objects_picked_up,
            "wandb_episode_end": {
                "fetch_reward_total": self._fetch_reward_total,
            }
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = fetch_reward + healthy_reward - costs

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

    def get_collisions(self) -> List[str]:
        """ Returns a list of objects (as object names) that the agent is currently touching. """
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

    def place_object(self, object_name, position=None, orientation=None, height=None):
        """ Places an object at a given position and orientation. If no position or orientation is given, the object is placed at a random location. """
        if position is None:
            position = np.random.uniform(low=1, high=4, size=2) * self._cell_size
        if height is None:
            height = np.random.uniform(low=3, high=10)
        if orientation is None:
            # Generate a random quaternion
            orientation = np.random.uniform(low=-1, high=1, size=4)
        # TODO: Look for the the joint associated with the body instead of guessing at the name
        adr = self.model.joint(f'{object_name}_joint').qposadr.item()
        self.data.qpos[adr:adr+2] = position
        self.data.qpos[adr+2] = height
        self.data.qpos[adr+3:adr+7] = orientation
