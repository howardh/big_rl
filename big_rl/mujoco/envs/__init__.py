from importlib.resources import files
from typing import Tuple

from bs4 import BeautifulSoup
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import tempfile

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


def create_mjcf_model():
    model_name = 'ant'

    soup = BeautifulSoup(features='xml')

    root = soup.new_tag('mujoco', attrs={'model': model_name})
    soup.append(root)

    root.append(soup.new_tag('compiler', attrs={
        'angle': 'degree',
        'coordinate': 'local',
        'inertiafromgeom': 'true',
    }))
    root.append(soup.new_tag('option', attrs={
        'integrator': 'RK4',
        'timestep': '0.01'
    }))

    # Custom
    custom = soup.new_tag('custom')
    custom.append(soup.new_tag('numeric', attrs={
        'data': '0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0',
        'name': 'init_qpos',
    }))
    root.append(custom)

    # Defaults
    default = soup.new_tag('default')
    default.append(soup.new_tag('joint', attrs={
        'armature': '1',
        'damping': '1',
        'limited': 'true'
    }))
    default.append(soup.new_tag('geom', attrs={
        'conaffinity': '0',
        'condim': '3',
        'density': '5',
        'friction': '1 0.5 0.5',
        'margin': '0.01',
        'rgba': '0.8 0.6 0.4 1',
    }))
    root.append(default)

    # Assets
    asset = soup.new_tag('asset')
    asset.append(soup.new_tag('texture', attrs={
        'builtin': 'gradient',
        'height': '100',
        'rgb1': '1 1 1',
        'rgb2': '0 0 0',
        'type': 'skybox',
        'width': '100'
    }))
    asset.append(soup.new_tag('texture', attrs={
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
    asset.append(soup.new_tag('texture', attrs={
        'builtin': 'checker',
        'height': '100',
        'name': 'texplane',
        'rgb1': '0 0 0',
        'rgb2': '0.8 0.8 0.8',
        'type': '2d',
        'width': '100'
    }))
    asset.append(soup.new_tag('material', attrs={
        'name': 'MatPlane',
        'reflectance': '0.5',
        'shininess': '1',
        'specular': '1',
        'texrepeat': '60 60',
        'texture': 'texplane'
    }))
    asset.append(soup.new_tag('material', attrs={
        'name': 'geom',
        'texture': 'texgeom',
        'texuniform': 'true'
    }))
    root.append(asset)

    # Worldbody
    worldbody = soup.new_tag('worldbody')
    root.append(worldbody)

    # Light
    worldbody.append(soup.new_tag('light', attrs={
        'cutoff': '100',
        'diffuse': '1 1 1',
        'dir': '0 0 -1.3',
        'directional': 'true',
        'exponent': '1',
        'pos': '0 0 1.3',
        'specular': '.1 .1 .1',
    }))

    # Ground
    worldbody.append(soup.new_tag('geom', attrs={
        'conaffinity': '1',
        'condim': '3',
        'material': 'MatPlane',
        'name': 'floor',
        'pos': '0 0 0',
        'rgba': '0.8 0.9 0.8 1',
        'size': '40 40 40',
        'type': 'plane',
    }))

    # Ant
    ant = create_mjcf_ant(soup)
    worldbody.append(ant['body'])

    # Walls
    walls = create_mjcf_boundary(soup, height=3, rgba=(1,0,0,0.3))
    worldbody.append(walls['body'])

    # Ball
    ball = create_mjcf_ball(soup, size=0.5, rgba=(0,0,1,1), name_prefix='obj1')
    worldbody.append(ball['body'])
    ball = create_mjcf_ball(soup, size=0.4, rgba=(1,0,0,1), name_prefix='obj2')
    worldbody.append(ball['body'])
    ball = create_mjcf_ball(soup, size=0.3, rgba=(0,1,0,1), name_prefix='obj3')
    worldbody.append(ball['body'])

    actuator = soup.new_tag('actuator')
    for a in ant['actuators']:
        actuator.append(a)
    root.append(actuator)

    return soup


def create_mjcf_ant(soup: BeautifulSoup, name_prefix: str = '', pos: Tuple[float, float, float] = (0, 0, 0.75), show_nose: bool = False):
    """
    <body name="torso" pos="0 0 0.75">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <camera name="first_person" mode="fixed" pos="0 0 0" axisangle="1 0 0 90"/>
      <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="aux_1_geom" size="0.08" type="capsule"/>
        <body name="aux_1" pos="0.2 0.2 0">
          <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="left_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.2 0.2 0">
            <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.4 0.4 0.0" name="left_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="aux_2_geom" size="0.08" type="capsule"/>
        <body name="aux_2" pos="-0.2 0.2 0">
          <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="right_leg_geom" size="0.08" type="capsule"/>
          <body pos="-0.2 0.2 0">
            <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="right_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="aux_3_geom" size="0.08" type="capsule"/>
        <body name="aux_3" pos="-0.2 -0.2 0">
          <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="back_leg_geom" size="0.08" type="capsule"/>
          <body pos="-0.2 -0.2 0">
            <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="third_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="right_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="aux_4_geom" size="0.08" type="capsule"/>
        <body name="aux_4" pos="0.2 -0.2 0">
          <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="rightback_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.2 -0.2 0">
            <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.4 -0.4 0.0" name="fourth_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
    """
    
    body = soup.new_tag('body', attrs={'name': 'torso', 'pos': ' '.join(map(str, pos))})
    actuators = []

    camera_track = soup.new_tag(
        'camera',
        attrs={
            'name': name_prefix+'track',
            'mode': 'trackcom',
            'pos': '0 -3 0.3',
            'xyaxes': '1 0 0 0 0 1'
        }
    )
    body.append(camera_track)

    camera_first_person = soup.new_tag('camera', attrs={'name': name_prefix+'first_person', 'mode': 'fixed', 'pos': '0 0 0', 'axisangle': '1 0 0 90'})
    body.append(camera_first_person)

    torso_geom = soup.new_tag('geom', attrs={'name': name_prefix+'torso_geom', 'pos': '0 0 0', 'size': '0.25', 'type': 'sphere'})
    body.append(torso_geom)

    # Make it a free joint so that we can move it around
    root_joint = soup.new_tag('joint', attrs={'armature': '0', 'damping': '0', 'limited': 'false', 'margin': '0.01', 'name': name_prefix+'root_joint', 'pos': '0 0 0', 'type': 'free'})
    body.append(root_joint)

    # Nose (for debugging purposes. Shows the direction of the first person camera.)
    if show_nose:
        body.append(soup.new_tag('geom', attrs={
            'contype': '0',
            'conaffinity': '0',
            'name': name_prefix+'nose_geom',
            'fromto': '0 0 0.1 0 10 0',
            'size': '0.01',
            'type': 'capsule',
            'rgba': '1 0 0 0.1',
            'mass': '0',
        }))

    # Legs
    for i in range(4):
        leg = soup.new_tag('body', attrs={
            'name': name_prefix+'leg_'+str(i),
            'pos': '0 0 0',
            'axisangle': f'0 0 1 {90*i}'
        })
        body.append(leg)

        leg_aux_geom = soup.new_tag('geom', attrs={
            'fromto': '0.0 0.0 0.0 0.2 0.2 0.0',
            'name': name_prefix+'leg_aux_geom_'+str(i),
            'size': '0.08',
            'type': 'capsule'
        })
        leg.append(leg_aux_geom)

        leg_aux = soup.new_tag('body', attrs={
            'name': name_prefix+'leg_aux_'+str(i),
            'pos': '0.2 0.2 0'
        })
        leg.append(leg_aux)

        leg_hip = soup.new_tag('joint', attrs={
            'axis': f'0 0 1',
            'name': name_prefix+'hip_'+str(i),
            'pos': '0.0 0.0 0.0',
            'range': '-30 30',
            'type': 'hinge'
        })
        leg_aux.append(leg_hip)
        actuators.append(soup.new_tag('motor', attrs={
            'ctrllimited': 'true',
            'ctrlrange': '-1 1',
            'joint': name_prefix+'hip_'+str(i),
            'gear': '150',
        }))

        leg_geom = soup.new_tag('geom', attrs={
            'fromto': '0.0 0.0 0.0 0.2 0.2 0.0',
            'name': name_prefix+'leg_geom_'+str(i),
            'size': '0.08',
            'type': 'capsule'
        })
        leg_aux.append(leg_geom)

        leg_body = soup.new_tag('body', attrs={'pos': '0.2 0.2 0'})
        leg_aux.append(leg_body)

        leg_ankle = soup.new_tag('joint', attrs={
            'axis': '-1 1 0',
            'name': name_prefix+'ankle_'+str(i),
            'pos': '0.0 0.0 0.0',
            'range': '30 70',
            'type': 'hinge'
        })
        leg_body.append(leg_ankle)
        actuators.append(soup.new_tag('motor', attrs={
            'ctrllimited': 'true',
            'ctrlrange': '-1 1',
            'joint': name_prefix+'ankle_'+str(i),
            'gear': '150',
        }))

        leg_ankle_geom = soup.new_tag('geom', attrs={
            'fromto': '0.0 0.0 0.0 0.4 0.4 0.0',
            'name': name_prefix+'ankle_geom_'+str(i),
            'size': '0.08',
            'type': 'capsule'
        })
        leg_body.append(leg_ankle_geom)

    return { 'body': body, 'actuators': actuators }


def create_mjcf_boundary(soup: BeautifulSoup, name_prefix: str = '', size: float = 3.0, thickness: float = 0.1, rgba=(1,0,0,0.5), height: float = 1.5):
    body = soup.new_tag('body', attrs={'name': name_prefix+'boundary', 'pos': f'0 0 {height/2}'})

    for i,(a,b) in enumerate([(1,0), (-1,0), (0,1), (0,-1)]):
        body.append(soup.new_tag('geom', attrs={
            'conaffinity': '1',
            'condim': '3',
            'material': 'MatPlane',
            'name': name_prefix+'wall'+str(i),
            'pos': f'{size*a} {size*b} 0',
            'size': f'{size if a==0 else thickness} {size if b==0 else thickness} {height}',
            'type': 'box',
            'rgba': ' '.join(map(str, rgba))
        }))
    body.append(soup.new_tag('geom', attrs={
        'conaffinity': '1',
        'condim': '3',
        'material': 'MatPlane',
        'name': name_prefix+'wall4',
        'pos': f'0 0 {height}',
        'size': f'{size} {size} {thickness}',
        'type': 'box',
        'rgba': ' '.join(map(str, rgba))
    }))

    return { 'body': body }


def create_mjcf_ball(soup: BeautifulSoup, rgba=(0,0,1,1), name_prefix: str = '', pos: Tuple[float] = (0,0,2), size: float = 0.1):
    body = soup.new_tag('body', attrs={'name': name_prefix+'ball', 'pos': ' '.join(map(str, pos))})

    body.append(soup.new_tag('joint', attrs={
        'armature': '0',
        'damping': '0',
        'limited': 'false',
        'margin': '0.01',
        'name': name_prefix+'ball_root_joint',
        'pos': '0 0 0',
        'type': 'free'
    }))
    body.append(soup.new_tag('geom', attrs={
        'conaffinity': '1',
        'name': name_prefix+'ball_geom',
        'pos': '0 0 0',
        'size': str(size),
        'type': 'sphere',
        'rgba': ' '.join(map(str, rgba))
    }))

    return { 'body': body }


class TestEnv(MujocoEnv, utils.EzPickle):
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
        xml_file=str(files('big_rl.mujoco.envs.assets').joinpath('ant.xml')),
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
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

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        obs_shape = 27
        if not exclude_current_positions_from_observation:
            obs_shape += 2
        if use_contact_forces:
            obs_shape += 84

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=observation_space,
            #default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

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
        raw_contact_forces = self.data.cfrc_ext
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
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

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

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._use_contact_forces:
            contact_force = self.contact_forces.flat.copy()
            return np.concatenate((position, velocity, contact_force))
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


register(
    id="TestEnv-v0",
    entry_point="big_rl.mujoco.envs:TestEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)


if __name__ == '__main__':
    ## See https://setuptools.pypa.io/en/latest/userguide/datafiles.html
    #data_text = files('big_rl.mujoco.envs.assets').joinpath('ant.xml').read_text()
    #print(data_text)

    #data_path = files('big_rl.mujoco.envs.assets').joinpath('ant.xml')
    #print(data_path)

    #env = gym.make('TestEnv-v0', render_mode='human')

    #env.reset()
    #for _ in range(1000):
    #    env.render()
    #    env.step(env.action_space.sample())

    soup = create_mjcf_model()
    print(soup.prettify())

    # save to a temp file
    with tempfile.NamedTemporaryFile(suffix='.xml') as f:
        f.write(soup.prettify().encode('utf-8'))
        f.flush()

        #print(f.name)
        #import subprocess
        #subprocess.run(['/home/howard/.mujoco/mujoco210/bin/simulate', f.name])

        env = gym.make('TestEnv-v0', render_mode='human', xml_file=f.name)
        env.reset()
        for _ in range(1000):
            env.render()
            env.step(env.action_space.sample())
