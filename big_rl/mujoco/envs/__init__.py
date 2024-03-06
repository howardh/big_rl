import os
import threading
import time
from typing import TypeVar, Optional

from bs4 import BeautifulSoup
import gymnasium
import gymnasium as gym
from gymnasium.spaces import Dict
from gymnasium.envs.registration import register
from gymnasium.wrappers import RecordEpisodeStatistics, ClipAction, TransformObservation, NormalizeReward, TransformReward # pyright: ignore[reportPrivateImportUsage]
from gymnasium.wrappers.normalize import RunningMeanStd
import numpy as np
import scipy.ndimage


def make_env(env_name: str,
        config={},
        #mujoco_config={},
        meta_config=None,
        normalize_obs=True,
        discount=0.99, reward_scale=1.0, reward_clip=10) -> gym.Env:
    env = gym.make(
            env_name, 
            **{'render_mode': 'rgb_array', **config},
    )

    env = RewardInfo(env)
    env = RecordEpisodeStatistics(env)

    env = ClipAction(env)

    if normalize_obs:
        env = NormalizeDictObservation(env)
    env = TransformObservation(env, lambda obs: {
        k: np.clip(v, -10, 10) if np.issubdtype(v.dtype, float) else v
        for k,v in obs.items()
    })

    env = NormalizeReward(env, gamma=discount)
    env = TransformReward(env, lambda reward: np.clip(reward * reward_scale, -reward_clip, reward_clip))

    if meta_config is not None:
        env = MetaWrapper(env, **meta_config)
    return env


ObsType = TypeVar('ObsType')
ActionType = TypeVar('ActionType')


class NormalizeDictObservation(gym.Wrapper[dict, ActionType, dict, ActionType]):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env[dict, ActionType], epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env)
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise TypeError("This wrapper only works with Dict observation spaces.")
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = {
                    k: RunningMeanStd(shape=obs_space.shape)
                    for k,obs_space in self.single_observation_space.items()
            }
            self.single_observation_space = gymnasium.spaces.Dict({
                    k: gymnasium.spaces.Box(-np.inf, np.inf, v.shape, dtype=np.float32) if isinstance(v, gymnasium.spaces.Box) else v
                    for k,v in self.single_observation_space.items()
            })
        else:
            self.obs_rms = {
                    k: RunningMeanStd(shape=obs_space.shape)
                    for k,obs_space in self.observation_space.items()
            }
            self.observation_space = Dict({
                    k: gymnasium.spaces.Box(-np.inf, np.inf, v.shape, dtype=np.float32) if isinstance(v, gymnasium.spaces.Box) else v
                    for k,v in self.observation_space.items()
            })
        self.epsilon = epsilon

    def step(self, action: ActionType):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action) # type: ignore
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = {
                    k: v[0]
                    for k,v in self.normalize({k: np.array([v]) for k,v in obs.items()}).items() # type: ignore
            }
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return {
                    k: v[0]
                    for k,v in self.normalize({k: np.array([v]) for k,v in obs.items()}).items() # type: ignore
            }, info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        for k,v in obs.items():
            self.obs_rms[k].update(v)
        return {
            k: (v - self.obs_rms[k].mean) / np.sqrt(self.obs_rms[k].var + self.epsilon)
            for k,v in obs.items()
        }


class RewardInfo(gym.Wrapper):
    """ Wrapper to add the reward to the wrapper. Used to save the true reward when the reward is manipulated down the line (e.g. via clipping, normalization, etc). """
    def __init__(self, env: gym.Env, key='reward'):
        super().__init__(env)
        self.key = key

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action) # type: ignore
        info[self.key] = reward
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info[self.key] = 0.0
        return obs, info


class MetaWrapper(gym.Wrapper):
    """
    Wrapper for meta-RL.

    Features:
    - Converting observations to dict
    - Adding reward, termination signal, and previous action to observations
    - Stacking episodes
    - Randomizing the environment between trials (requires the environment to have a `randomize()` method)
    """
    def __init__(self,
            env,
            episode_stack: int,
            dict_obs: bool = False,
            randomize: bool = True,
            #action_shuffle: bool = False,
            #include_action_map: bool = False,
            include_reward: bool = True,
            image_transformation = None,
            task_id = None,
            task_label = None,
            seed: Optional[int] = None):
        super().__init__(env)
        self.episode_stack = episode_stack
        self.randomize = randomize
        self.dict_obs = dict_obs
        #self.action_shuffle = action_shuffle
        #self.include_action_map = include_action_map
        self.include_reward = include_reward
        self.task_id = task_id
        self.task_label = task_label

        self.episode_count = 0
        self._done = True

        if seed is None:
            seed = os.getpid() + int(time.time())
            thread_id = threading.current_thread().ident
            if thread_id is not None:
                seed += thread_id
            seed = seed % (2**32 - 1)
        self.rand = np.random.RandomState(seed)

        #if action_shuffle:
        #    self._randomize_actions()
        #if self.include_action_map:
        #    assert self.action_shuffle, '`action_shuffle` must be enabled along with `include_action_map`.'
        #    assert self.dict_obs, '`dict_obs` must be enabled along with `include_action_map`.'

        self._transform = lambda x: x
        if image_transformation is not None:
            if self.rand.rand() < image_transformation.get('vflip', 0):
                self._transform = lambda img, f=self._transform: f(img[:,::-1,:])
            if self.rand.rand() < image_transformation.get('hflip', 0):
                self._transform = lambda img, f=self._transform: f(img[:,:,::-1])
            if self.rand.rand() < image_transformation.get('rotate', 0):
                angle = (self.rand.rand() - 0.5) * (3.141592653 * 2) * (4 / 360)
                self._transform = lambda img, f=self._transform: f(
                        scipy.ndimage.rotate(img, angle, reshape=False))
            self._transform = lambda obs, f=self._transform: {'image': f(obs['image']), **obs}

        if dict_obs:
            obs_space = [('obs', self.env.observation_space)]
            if isinstance(self.env.observation_space, gymnasium.spaces.Dict):
                obs_space = [(f'obs ({k})',v) for k,v in self.env.observation_space.items()]
            obs_space = [
                ('done', gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)),
                *obs_space,
                ('action', self.env.action_space),
            ]
            if self.include_reward:
                obs_space.append(
                    ('reward', gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32))
                )
            #if self.include_action_map:
            #    assert isinstance(self.env.action_space, gymnasium.spaces.Discrete)
            #    obs_space.append((
            #        'action_map',
            #        gymnasium.spaces.Box(
            #            low=0, high=1,
            #            shape=(self.env.action_space.n, self.env.action_space.n),
            #            dtype=np.float32
            #        )
            #    ))
            self.observation_space = gymnasium.spaces.Dict(obs_space)

    def _randomize_actions(self):
        env = self.env
        assert isinstance(env.action_space, gymnasium.spaces.Discrete), 'Action shuffle only works with discrete actions'
        n = env.action_space.n
        self.action_map = np.arange(n)
        self.rand.shuffle(self.action_map)

        self.action_map_obs = np.zeros([n,n])
        for i,a in enumerate(self.action_map):
            self.action_map_obs[i,a] = 1

    def step(self, action):
        # Map action to the shuffled action space
        original_action = action
        #if self.action_shuffle:
        #    action = self.action_map[action]

        # Take a step
        if self._done:
            if self.episode_count == 0 and self.randomize:
                self.env.randomize() # type: ignore
            self.episode_count += 1
            self._done = False
            (obs, info), reward, terminated, truncated = self.env.reset(), 0, False, False
        else:
            obs, reward, terminated, truncated, info = self.env.step(action) # type: ignore
        done = terminated or truncated
        self._done = done

        # Add task ID
        if self.task_id is not None:
            info['task_id'] = self.task_id
        if self.task_label is not None:
            info['task_label'] = self.task_label

        # Apply transformations
        obs = self._transform(obs)

        # Convert observation to dict
        if self.dict_obs:
            if isinstance(self.env.observation_space, gymnasium.spaces.Dict):
                obs = {
                    'done': np.array([done], dtype=np.float32),
                    **{f'obs ({k})': v for k,v in obs.items()},
                    'action': original_action,
                }
            else:
                obs = {
                    'done': np.array([done], dtype=np.float32),
                    'obs': obs,
                    'action': original_action,
                }
            if self.include_reward:
                obs['reward'] = np.array([info.get('reward',reward)], dtype=np.float32)
            #if self.include_action_map:
            #    obs['action_map'] = self.action_map_obs

        if done:
            if 'expected_return' in info and 'max_return' in info:
                regret = info['max_return'] - info['expected_return']
                self._regret.append(regret)
                info['regret'] = self._regret

            if self.episode_count >= self.episode_stack: # Episode count starts at 1
                self.episode_count = 0
                return obs, reward, terminated, truncated, info
            else:
                return obs, reward, False, False, info
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.episode_count = 0
        if self.randomize:
            self.env.randomize() # type: ignore
        #if self.action_shuffle:
        #    self._randomize_actions()

        self._regret = []

        obs, info = self.env.reset(seed=seed, options=options)

        obs = self._transform(obs)
        if self.dict_obs:
            if isinstance(self.env.observation_space, gymnasium.spaces.Dict):
                obs = {
                    'done': np.array([False], dtype=np.float32),
                    **{f'obs ({k})': v for k,v in obs.items()},
                    'action': self.env.action_space.sample(),
                }
            else:
                obs = {
                    'done': np.array([False], dtype=np.float32),
                    'obs': obs,
                    'action': self.env.action_space.sample(),
                }
            if self.include_reward:
                obs['reward'] = np.array([0], dtype=np.float32)
            #if self.include_action_map:
            #    obs['action_map'] = self.action_map_obs

        # Add task id
        if self.task_id is not None:
            info['task_id'] = self.task_id
        if self.task_label is not None:
            info['task_label'] = self.task_label

        return obs, info


DEFAULT_CAMERA_CONFIG = { # See mjvCamera docs (https://mujoco.readthedocs.io/en/latest/APIreference.html#mjvcamera)
    "distance": 4.0,
    #"fixedcamid": 1,
    #"trackbodyid": 1,
    #"lookat": [0.0, 0.0, 0.0],
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

    def add_light(self, pos=(0, 0, 5)):
        self.worldbody.append(self.soup.new_tag('light', attrs={
            'cutoff': '100',
            'diffuse': '1 1 1',
            'dir': '0 0 -1',
            'exponent': '1',
            'pos': f'{pos[0]} {pos[1]} {pos[2]}',
            'specular': '1 1 1'
        }))

    def add_ant(self, pos, name_prefix='', show_nose = False, num_legs=4):
        body_name = f'{name_prefix}torso'
        body = self.soup.new_tag('body', attrs={
            'name': body_name,
            'pos': f'{pos[0] * self.cell_size} {pos[1] * self.cell_size} 0.75'
        })
        actuators = []

        # Side camera
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

        # Top-down camera
        camera_track = self.soup.new_tag(
            'camera',
            attrs={
                'name': name_prefix+'topdown',
                'mode': 'trackcom',
                'pos': '0 0 10',
                'xyaxes': '1 0 0 0 1 0'
            }
        )
        body.append(camera_track)

        # First person camera
        camera_first_person = self.soup.new_tag('camera', attrs={'name': name_prefix+'first_person', 'mode': 'fixed', 'pos': '0 0 0', 'axisangle': '1 0 0 90'})
        body.append(camera_first_person)

        # Torso
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

    def add_arm(self, pos, name_prefix='', segment_lengths=[2, 2], camera_fov=45, camera_distance=1, gear=[150,150,150]):
        tip_length = 0.1

        body_name = f'{name_prefix}torso'
        body = self.soup.new_tag('body', attrs={
            'name': body_name,
            'pos': f'{pos[0] * self.cell_size} {pos[1] * self.cell_size} 0.75'
        })
        actuators = []

        # Top-down camera
        camera_track = self.soup.new_tag(
            'camera',
            attrs={
                'name': name_prefix+'topdown',
                'mode': 'track',
                'pos': '0 0 10',
                'xyaxes': '1 0 0 0 1 0',
                #'fovy': '135',
            }
        )
        body.append(camera_track)

        # Base
        base_geom = self.soup.new_tag('geom', attrs={
            'name': name_prefix+'base_geom',
            'fromto': '0 0 0 0 0 0.5',
            'size': '0.25',
            'type': 'capsule',
            'rgba': '0.3 0.3 0.3 1.0',
        })
        body.append(base_geom)

        # Base can rotate around the vertical axis
        root_joint = self.soup.new_tag('joint', attrs={
            'axis': '0 0 1',
            'name': name_prefix+'root_joint',
            'pos': '0 0 0',
            'range': '-180 180',
            'type': 'hinge'
        })
        body.append(root_joint)
        actuators.append(self.soup.new_tag('motor', attrs={
            'ctrllimited': 'true',
            'ctrlrange': '-1 1',
            'joint': name_prefix+'root_joint',
            'gear': str(gear[0]),
        }))


        # Arm segment 1
        arm_segment_1 = self.soup.new_tag('body', attrs={'name': name_prefix+'arm_segment_1', 'pos': '0 0 0.5'})
        body.append(arm_segment_1)

        arm_segment_1_geom = self.soup.new_tag('geom', attrs={
            'name': name_prefix+'arm_segment_1_geom',
            'fromto': f'0 0 0 0 0 {segment_lengths[0]}',
            'size': '0.25',
            'type': 'capsule',
            'rgba': '0.5 0.5 0.5 1.0',
        })
        arm_segment_1.append(arm_segment_1_geom)

        arm_segment_1_joint = self.soup.new_tag('joint', attrs={
            'axis': '1 0 0',
            'name': name_prefix+'arm_segment_1_joint',
            'pos': '0 0 0',
            'range': '-45 90',
            'type': 'hinge'
        })
        arm_segment_1.append(arm_segment_1_joint)
        actuators.append(self.soup.new_tag('motor', attrs={
            'ctrllimited': 'true',
            'ctrlrange': '-1 1',
            'joint': name_prefix+'arm_segment_1_joint',
            'gear': str(gear[1]),
        }))

        # Arm segment 2
        arm_segment_2 = self.soup.new_tag('body', attrs={'name': name_prefix+'arm_segment_2', 'pos': f'0 0 {segment_lengths[0]}'})
        arm_segment_1.append(arm_segment_2)

        arm_segment_2_geom = self.soup.new_tag('geom', attrs={
            'name': name_prefix+'arm_segment_2_geom',
            'fromto': f'0 0 0 0 0 {segment_lengths[1]-tip_length}',
            'size': '0.25',
            'type': 'capsule',
            'rgba': '0.3 0.3 0.3 1.0',
        })
        arm_segment_2.append(arm_segment_2_geom)

        arm_segment_2_joint = self.soup.new_tag('joint', attrs={
            'axis': '1 0 0',
            'name': name_prefix+'arm_segment_2_joint',
            'pos': '0 0 0',
            'range': '0 135',
            'type': 'hinge'
        })
        arm_segment_2.append(arm_segment_2_joint)
        actuators.append(self.soup.new_tag('motor', attrs={
            'ctrllimited': 'true',
            'ctrlrange': '-1 1',
            'joint': name_prefix+'arm_segment_2_joint',
            'gear': str(gear[2]),
        }))

        # First person camera
        camera_first_person = self.soup.new_tag('camera', attrs={
            'name': name_prefix+'first_person',
            'mode': 'fixed',
            'pos': f'0 0 {segment_lengths[1]*camera_distance}',
            'axisangle': '0 1 0 180',
            'fovy': str(camera_fov),
        })
        arm_segment_2.append(camera_first_person)

        # Effector tip
        arm_tip = self.soup.new_tag('body', attrs={'name': name_prefix+'arm_tip', 'pos': f'0 0 {segment_lengths[1]}'})
        arm_segment_2.append(arm_tip)

        arm_tip_geom = self.soup.new_tag('geom', attrs={
            'name': name_prefix+'arm_tip_geom',
            'fromto': f'0 0 0 0 0 0.1',
            'size': '0.25',
            'type': 'capsule',
            'rgba': '1 0 0 1',
        })
        arm_tip.append(arm_tip_geom)

        self.worldbody.append(body)
        for a in actuators:
            self.actuator.append(a)
        self.agent = { 'body': body, 'actuators': actuators }

        return body_name

    def add_wall(self, cell, cell2=None, rgba=(0.8, 0.8, 0.8, 1), height=None):
        if height is None:
            height = self.height
        if cell2 is None:
            pos = (
                cell[0] * self.cell_size,
                cell[1] * self.cell_size,
                height / 2,
            )
            size = (self.cell_size / 2, self.cell_size / 2, height / 2)
        else:
            pos = (
                (cell[0] + cell2[0]) * self.cell_size / 2,
                (cell[1] + cell2[1]) * self.cell_size / 2,
                height / 2,
            )
            size = (
                (abs(cell[0] - cell2[0]) + 1) * self.cell_size / 2,
                (abs(cell[1] - cell2[1]) + 1) * self.cell_size / 2,
                height / 2,
            )
        wall = self.soup.new_tag('geom', attrs={
            'conaffinity': '1',
            'condim': '3',
            'material': 'MatPlane',
            'name': f'wall_{len(self.walls)}',
            'pos': ' '.join([str(p) for p in pos]),
            'size': ' '.join([str(s) for s in size]),
            'type': 'box',
            'rgba': ' '.join([str(c) for c in rgba]),
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

        self.boxes.append(body)
        self.worldbody.append(body)

        return body_name

    def set_boundary(self, x_min, x_max, y_min, y_max, height=10):
        """
        Set the boundary of the world. The provided coordinates are all in cell coordinates. A 1 cell thick invisible wall will be placed on the provided coordinates.
        """
        assert x_min < x_max
        assert y_min < y_max

        rgba = (0, 0, 0, 0)
        self.add_wall(
                (x_min, y_min), (x_min, y_max),
                rgba = rgba,
                height = height * self.cell_size
        )
        self.add_wall(
                (x_min, y_max), (x_max, y_max),
                rgba = rgba,
                height = height * self.cell_size
        )
        self.add_wall(
                (x_max, y_max), (x_max, y_min),
                rgba = rgba,
                height = height * self.cell_size
        )
        self.add_wall(
                (x_max, y_min), (x_min, y_min),
                rgba = rgba,
                height = height * self.cell_size
        )


def strz(x):
    """ Truncate a null-terminated string. """
    for i,c in enumerate(x):
        if c == 0:
            return x[:i]
    return x


def list_geoms_by_body(env, body_id=0, depth=0, covered_bodies=None):
    if depth == 0:
        print(f'Geoms by body:')

    if covered_bodies is None:
        covered_bodies = set()
    if body_id in covered_bodies:
        return
    covered_bodies.add(body_id)

    body_name_adr = env.model.name_bodyadr[body_id]
    body_name = strz(env.model.names[body_name_adr:])
    print(f'{"  "*(depth+1)} {body_id} {body_name}')
    for geom_id,geom_name_adr in enumerate(env.model.name_geomadr):
        geom_name = strz(env.model.names[geom_name_adr:])
        if env.model.geom_bodyid[geom_id] == body_id:
            print(f'{"  "*(depth+1)}   {geom_id} {geom_name}')
    for child_id,parent_id in enumerate(env.model.body_parentid):
        if parent_id == body_id:
            list_geoms_by_body(env, child_id, depth+1, covered_bodies)


def list_joints_with_ancestor(env, body_id, depth=0, covered_bodies=None, verbose=False):
    if depth == 0 and verbose:
        print(f'Joints with {body_id} as an ancestor:')
    joint_ids = set()

    if covered_bodies is None:
        covered_bodies = set()
    if body_id in covered_bodies:
        return joint_ids
    covered_bodies.add(body_id)

    body_name_adr = env.model.name_bodyadr[body_id]
    body_name = strz(env.model.names[body_name_adr:])
    if verbose:
        print(f'{"  "*(depth+1)} {body_id} {body_name}')
    for joint_id,joint_name_adr in enumerate(env.model.name_jntadr):
        joint_name = strz(env.model.names[joint_name_adr:])
        if env.model.jnt_bodyid[joint_id] == body_id:
            if verbose:
                print(f'{"  "*(depth+1)}   {joint_id} {joint_name}')
            joint_ids.add(joint_id)
    for child_id,parent_id in enumerate(env.model.body_parentid):
        if parent_id == body_id:
            j = list_joints_with_ancestor(env, child_id, depth+1, covered_bodies)
            joint_ids.update(j)
    return joint_ids


def list_bodies_with_ancestor(env, body_id, depth=0, verbose=False):
    if depth == 0 and verbose:
        print(f'Bodies with {body_id} as an ancestor:')
    body_ids = set([body_id])

    body_name_adr = env.model.name_bodyadr[body_id]
    body_name = strz(env.model.names[body_name_adr:])
    if verbose:
        print(f'{"  "*(depth+1)} {body_id} {body_name}')
    for child_id,parent_id in enumerate(env.model.body_parentid):
        if parent_id == body_id:
            body_ids.update(list_bodies_with_ancestor(env, child_id, depth+1))
    return body_ids


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return np.array([
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    ])


def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by a quaternion.

    Args:
        v: A 3D vector (vx,xy,xz).
        q: A quaternion (qw,qx,qy,qz).
    """
    q_conj = quaternion_conjugate(q)
    return quaternion_multiply(quaternion_multiply(q, np.array([0, *v])), q_conj)[1:]


def is_upsidedown(q: np.ndarray, threshold: float = 0.) -> bool:
    """
    Check if a quaternion represents an upside-down orientation.
    The positive z-axis is considered to be the "up" direction.

    Args:
        q: A quaternion (qw,qx,qy,qz).
        threshold: The threshold for the dot product between the z-axis and the up direction.
    """
    v = rotate_vector_by_quaternion(np.array([0, 0, 1]), q)
    return v[2] < threshold


register(
    id="AntBaseline-v0",
    entry_point="big_rl.mujoco.envs.ant_baseline:AntBaselineEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="AntFetch-v0",
    entry_point="big_rl.mujoco.envs.ant_fetch:AntFetchEnv",
    #max_episode_steps=1000,
    #reward_threshold=6000.0,
)

register(
    id="ArmFetch-v0",
    entry_point="big_rl.mujoco.envs.arm_fetch:ArmFetchEnv",
)

register(
    id="HopperNoVelocity-v4",
    entry_point="big_rl.mujoco.envs.hopper_no_velocity:HopperNoVelocityEnv_v4",
)

register(
    id="HalfCheetahNoVelocity-v4",
    entry_point="big_rl.mujoco.envs.half_cheetah_no_velocity:HalfCheetahNoVelocityEnv_v4",
)

register(
    id="AntNoVelocity-v4",
    entry_point="big_rl.mujoco.envs.ant_no_velocity:AntNoVelocityEnv_v4",
)

register(
    id="AntForward-v4",
    entry_point="big_rl.mujoco.envs.ant_fw_bw:AntForwardBackwardEnv",
    kwargs={'target_direction': 1},
)

register(
    id="AntBackward-v4",
    entry_point="big_rl.mujoco.envs.ant_fw_bw:AntForwardBackwardEnv",
    kwargs={'target_direction': -1},
)

register(
    id="HalfCheetahForward-v4",
    entry_point="big_rl.mujoco.envs.half_cheetah_fw_bw:HalfCheetahForwardBackwardEnv",
    max_episode_steps=1000,
    kwargs={'target_direction': 1},
)

register(
    id="HalfCheetahBackward-v4",
    entry_point="big_rl.mujoco.envs.half_cheetah_fw_bw:HalfCheetahForwardBackwardEnv",
    max_episode_steps=1000,
    kwargs={'target_direction': -1},
)

register(
    id="HalfCheetahNoVelocityForward-v4",
    entry_point="big_rl.mujoco.envs.half_cheetah_fw_bw:HalfCheetahForwardBackwardEnv",
    max_episode_steps=1000,
    kwargs={'target_direction': 1, 'exclude_velocity_from_observation': True},
)

register(
    id="HalfCheetahNoVelocityBackward-v4",
    entry_point="big_rl.mujoco.envs.half_cheetah_fw_bw:HalfCheetahForwardBackwardEnv",
    max_episode_steps=1000,
    kwargs={'target_direction': -1, 'exclude_velocity_from_observation': True},
)

register(
    id="SwimmerForward-v4",
    entry_point="big_rl.mujoco.envs.swimmer_fw_bw:SwimmerForwardBackwardEnv",
    kwargs={'target_direction': 1},
)

register(
    id="SwimmerBackward-v4",
    entry_point="big_rl.mujoco.envs.swimmer_fw_bw:SwimmerForwardBackwardEnv",
    kwargs={'target_direction': -1},
)

register(
    id="Walker2dForward-v4",
    entry_point="big_rl.mujoco.envs.walker2d_fw_bw:Walker2dForwardBackwardEnv",
    kwargs={'target_direction': 1},
)

register(
    id="Walker2dBackward-v4",
    entry_point="big_rl.mujoco.envs.walker2d_fw_bw:Walker2dForwardBackwardEnv",
    kwargs={'target_direction': -1},
)

register(
    id="HumanoidForward-v4",
    entry_point="big_rl.mujoco.envs.humanoid_fw_bw:HumanoidForwardBackwardEnv",
    kwargs={'target_direction': 1},
)

register(
    id="HumanoidBackward-v4",
    entry_point="big_rl.mujoco.envs.humanoid_fw_bw:HumanoidForwardBackwardEnv",
    kwargs={'target_direction': -1},
)

register(
    id="HopperForward-v4",
    entry_point="big_rl.mujoco.envs.hopper_fw_bw:HopperForwardBackwardEnv",
    kwargs={'target_direction': 1},
)

register(
    id="HopperBackward-v4",
    entry_point="big_rl.mujoco.envs.hopper_fw_bw:HopperForwardBackwardEnv",
    kwargs={'target_direction': -1},
)

register(
    id="ReacherNoLimit-v4",
    entry_point="gymnasium.envs.mujoco.reacher_v4:ReacherEnv",
)

register(
    id="PusherNoLimit-v4",
    entry_point="gymnasium.envs.mujoco.pusher_v4:PusherEnv",
)


if __name__ == '__main__':
    env = gym.make('ArmFetch-v0', render_mode='human', num_objs=6, camera_fov=90, camera_distance=0)
    #env = gym.make('AntBaseline-v0', render_mode='human')
    env.reset()
    #breakpoint()
    print(env.action_space)
    total_reward = 0
    for i in range(1000):
        #breakpoint()
        #env.model.opt.timestep = 0.01 * (i%2+1) # Timestep size can be modified on the fly
        #print(f'{env.model.opt.timestep} {env.data.time:.2f}')
        #env.render()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        total_reward += float(reward)
        done = terminated or truncated
        if done:
            print(f'Episode terminated with reward {total_reward}')
            total_reward = 0
            obs, info = env.reset()

        ## Display image
        #import matplotlib
        #from matplotlib import pyplot as plt
        #plt.imshow(obs['image'].transpose(1,2,0))
        #plt.show()

        ## Save image
        #import imageio
        #imageio.imwrite(f'frame_{i:04d}.png', rgb_array)

        #breakpoint()
