import os
import copy
from typing import Mapping
import threading
import time

import numpy as np
import gymnasium as gym
import gymnasium.spaces
import gymnasium.utils.seeding
from gymnasium.envs.registration import register
from gymnasium.spaces import Box
import cv2
import scipy.ndimage
from permutation import Permutation
import minigrid.wrappers
import minigrid.core
import minigrid.core.mission
import minigrid.core.grid
import minigrid.core.constants
import minigrid.core.world_object
from minigrid.minigrid_env import MiniGridEnv, MissionSpace


def make_env(env_name: str,
        config={},
        minigrid_config={},
        meta_config=None,
        episode_stack=None,
        dict_obs=False,
        action_shuffle=False) -> gym.Env:
    env = gym.make(env_name, **config)
    env = MinigridPreprocessing(env, **minigrid_config)
    if episode_stack is not None:
        env = EpisodeStack(env, episode_stack, dict_obs=dict_obs)
    elif dict_obs:
        raise Exception('dict_obs requires episode_stack')
    if meta_config is not None:
        env = MetaWrapper(env, **meta_config)
    if action_shuffle:
        env = ActionShuffle(env)
    return env


class MinigridPreprocessing(gym.Wrapper):
    def __init__(
        self,
        env,
        rgb = True,
        screen_size = None,
        with_mission = False,
    ):
        super().__init__(env)
        assert ( cv2 is not None ), 'opencv-python package not installed!'

        if rgb:
            self.env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
            assert isinstance(self.env.observation_space, gymnasium.spaces.Dict)
            assert self.env.observation_space['image'].shape is not None
            assert isinstance(self.observation_space, gymnasium.spaces.Dict)
            self.observation_space['image'] = Box(
                    low=0, high=255,
                    shape=(
                        self.env.observation_space['image'].shape[2],
                        self.env.observation_space['image'].shape[0],
                        self.env.observation_space['image'].shape[1]),
                    dtype=np.uint8)

        self._with_mission = with_mission
        if not with_mission:
            assert isinstance(self.env.observation_space, gymnasium.spaces.Dict)
            self.observation_space = gymnasium.spaces.Dict({
                k:v for k,v in self.env.observation_space.items()
                if k != 'mission'
            })

        self.screen_size = screen_size
        self.rgb = rgb
    
    def _resize_obs(self, obs):
        if not self.rgb:
            return obs

        # Resize
        if self.screen_size is not None:
            obs['image'] = cv2.resize( # type: ignore
                obs['image'],
                (self.screen_size, self.screen_size),
                interpolation=cv2.INTER_AREA, # type: ignore
            )

        # Move channel dimension to start
        obs['image'] = np.moveaxis(obs['image'], 2, 0)

        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._resize_obs(obs)
        ## XXX: Check for reward permutation (until bug is mixed in minigrid)
        #if self.env.unwrapped.include_reward_permutation:
        #    obs['reward_permutation'] = self.env.unwrapped.reward_permutation
        if not self._with_mission:
            obs = {k:v for k,v in obs.items() if k != 'mission'}
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # NoopReset
        obs, info = self.env.reset(**kwargs)
        obs = self._resize_obs(obs)
        ## XXX: Check for reward permutation (until bug is mixed in minigrid)
        #if self.env.unwrapped.include_reward_permutation:
        #    obs['reward_permutation'] = self.env.unwrapped.reward_permutation
        if not self._with_mission:
            obs = {k:v for k,v in obs.items() if k != 'mission'}
        return obs, info


class EpisodeStack(gym.Wrapper):
    def __init__(self, env, num_episodes : int, dict_obs: bool = False):
        super().__init__(env)
        self.num_episodes = num_episodes
        self.episode_count = 0
        self.dict_obs = dict_obs
        self._done = True

        if dict_obs:
            obs_space = [('obs', self.env.observation_space)]
            if isinstance(self.env.observation_space, gymnasium.spaces.Dict):
                obs_space = [(f'obs ({k})',v) for k,v in self.env.observation_space.items()]
            self.observation_space = gymnasium.spaces.Dict([
                ('reward', gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)),
                ('done', gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)),
                #('obs', self.env.observation_space),
                *obs_space,
                ('action', self.env.action_space),
            ])

    def step(self, action):
        if self._done:
            if isinstance(self.env.unwrapped, NRoomBanditsSmall):
                self.env.unwrapped.shuffle_goals_on_reset = False
            self.episode_count += 1
            self._done = False
            (obs, info), reward, terminated, truncated = self.env.reset(), 0, False, False
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._done = done

        if self.dict_obs:
            if isinstance(self.env.observation_space, gymnasium.spaces.Dict):
                obs = {
                    'reward': np.array([reward], dtype=np.float32),
                    'done': np.array([done], dtype=np.float32),
                    **{f'obs ({k})': v for k,v in obs.items()},
                    'action': action,
                }
            else:
                obs = {
                    'reward': np.array([reward], dtype=np.float32),
                    'done': np.array([done], dtype=np.float32),
                    'obs': obs,
                    'action': action,
                }

        if done:
            if self.episode_count >= self.num_episodes:
                self.episode_count = 0
                return obs, reward, terminated, truncated, info
            else:
                return obs, reward, False, False, info
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.episode_count = 0
        if isinstance(self.env.unwrapped, NRoomBanditsSmall):
            self.env.unwrapped.shuffle_goals_on_reset = True
        obs, info = self.env.reset(**kwargs)
        if self.dict_obs:
            if isinstance(self.env.observation_space, gymnasium.spaces.Dict):
                return {
                    'reward': np.array([0], dtype=np.float32),
                    'done': np.array([False], dtype=np.float32),
                    **{f'obs ({k})': v for k,v in obs.items()},
                    'action': self.env.action_space.sample(),
                }, info
            else:
                return {
                    'reward': np.array([0], dtype=np.float32),
                    'done': np.array([False], dtype=np.float32),
                    'obs': obs,
                    'action': self.env.action_space.sample(),
                }, info
        else:
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
            action_shuffle: bool = False,
            include_action_map: bool = False,
            image_transformation = None,
            task_id = None,
            task_label = None,
            seed: int = None):
        super().__init__(env)
        self.episode_stack = episode_stack
        self.randomize = randomize
        self.dict_obs = dict_obs
        self.action_shuffle = action_shuffle
        self.include_action_map = include_action_map
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

        if action_shuffle:
            self._randomize_actions()
        if self.include_action_map:
            assert self.action_shuffle, '`action_shuffle` must be enabled along with `include_action_map`.'
            assert self.dict_obs, '`dict_obs` must be enabled along with `include_action_map`.'

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
                ('reward', gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)),
                ('done', gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)),
                *obs_space,
                ('action', self.env.action_space),
            ]
            if self.include_action_map:
                assert isinstance(self.env.action_space, gymnasium.spaces.Discrete)
                obs_space.append((
                    'action_map',
                    gymnasium.spaces.Box(
                        low=0, high=1,
                        shape=(self.env.action_space.n, self.env.action_space.n),
                        dtype=np.float32
                    )
                ))
            self.observation_space = gymnasium.spaces.Dict(obs_space)

    def _randomize_actions(self):
        env = self.env
        n = env.action_space.n
        assert isinstance(env.action_space, gymnasium.spaces.Discrete), 'Action shuffle only works with discrete actions'
        self.action_map = np.arange(n)
        self.rand.shuffle(self.action_map)

        self.action_map_obs = np.zeros([n,n])
        for i,a in enumerate(self.action_map):
            self.action_map_obs[i,a] = 1

    def step(self, action):
        # Map action to the shuffled action space
        original_action = action
        if self.action_shuffle:
            action = self.action_map[action]

        # Take a step
        if self._done:
            if self.episode_count == 0 and self.randomize:
                self.env.randomize()
            self.episode_count += 1
            self._done = False
            (obs, info), reward, terminated, truncated = self.env.reset(), 0, False, False
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
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
                    'reward': np.array([reward], dtype=np.float32),
                    'done': np.array([done], dtype=np.float32),
                    **{f'obs ({k})': v for k,v in obs.items()},
                    'action': original_action,
                }
            else:
                obs = {
                    'reward': np.array([reward], dtype=np.float32),
                    'done': np.array([done], dtype=np.float32),
                    'obs': obs,
                    'action': original_action,
                }
            if self.include_action_map:
                obs['action_map'] = self.action_map_obs

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

    def reset(self):
        self.episode_count = 0
        if self.randomize:
            self.env.randomize()
        if self.action_shuffle:
            self._randomize_actions()

        self._regret = []

        obs, info = self.env.reset()

        obs = self._transform(obs)
        if self.dict_obs:
            if isinstance(self.env.observation_space, gymnasium.spaces.Dict):
                obs = {
                    'reward': np.array([0], dtype=np.float32),
                    'done': np.array([False], dtype=np.float32),
                    **{f'obs ({k})': v for k,v in obs.items()},
                    'action': self.env.action_space.sample(),
                }
            else:
                obs = {
                    'reward': np.array([0], dtype=np.float32),
                    'done': np.array([False], dtype=np.float32),
                    'obs': obs,
                    'action': self.env.action_space.sample(),
                }
            if self.include_action_map:
                obs['action_map'] = self.action_map_obs

        # Add task id
        if self.task_id is not None:
            info['task_id'] = self.task_id
        if self.task_label is not None:
            info['task_label'] = self.task_label

        return obs, info


class ActionShuffle(gym.Wrapper):
    def __init__(self, env, actions=None, permutation=None):
        """
        Args:
            env: gym.Env
            actions: list of ints, indices of actions to shuffle. Alternatively, if set to True, then all actions are shuffled.
            permutation: list of ints, indices of the permutation to use. If not set, then a new permutation is randomly generated at the start of each episode.
        """
        super().__init__(env)
        if actions is None:
            self._actions = list(range(self.env.action_space.n))
        else:
            self._actions = actions

        if isinstance(permutation, list):
            self.permutation = permutation
        elif isinstance(permutation, int):
            self.permutation = Permutation.from_lehmer(permutation, len(self._actions)).to_image()
        else:
            self.permutation = None

        self._init_mapping()

    def _init_mapping(self):
        if isinstance(self.env.action_space, gymnasium.spaces.Discrete):
            self.mapping = np.arange(self.env.action_space.n)
            for i,j in enumerate(np.random.permutation(len(self._actions))):
                self.mapping[self._actions[i]] = self._actions[j]
        else:
            raise ValueError("ActionShuffle only supports Discrete action spaces")

    def reset(self, **kwargs):
        self._init_mapping()
        return self.env.reset(**kwargs)

    def step(self, action):
        action = self.mapping[action]
        return self.env.step(action)


def merge(source, destination):
    """
    (Source: https://stackoverflow.com/a/20666342/382388)

    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    destination = copy.deepcopy(destination)
    if isinstance(source, Mapping):
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                destination[key] = merge(value, node)
            elif isinstance(value, list):
                if isinstance(destination[key],list):
                    destination[key] = [merge(s,d) for s,d in zip(source[key],destination[key])]
                else:
                    destination[key] = value
            else:
                destination[key] = value

        return destination
    else:
        return source


class ExperimentConfigs(dict):
    def __init__(self):
        self._last_key = None
    def add(self, key, config, inherit=None):
        if key in self:
            raise Exception(f'Key {key} already exists.')
        if inherit is None:
            self[key] = config
        else:
            self[key] = merge(config,self[inherit])
        self._last_key = key
    def add_change(self, key, config):
        self.add(key, config, inherit=self._last_key)


class GoalDeterministic(minigrid.core.world_object.Goal):
    def __init__(self, reward):
        super().__init__()
        self.reward = reward


class GoalMultinomial(minigrid.core.world_object.Goal):
    def __init__(self, rewards, probs):
        super().__init__()
        self.rewards = rewards
        self.probs = probs

    def sample_reward(self):
        return self.rewards[np.random.choice(len(self.rewards), p=self.probs)]

    @property
    def expected_value(self):
        return (np.array(self.rewards, dtype=np.float32) * np.array(self.probs, dtype=np.float32)).sum()


default_mission_space = MissionSpace(
        mission_func=lambda: ""
)


class NRoomBanditsSmall(MiniGridEnv):
    def __init__(self, rewards=[-1,1], shuffle_goals_on_reset=True, include_reward_permutation=False, seed=None):
        self.mission = 'Reach the goal with the highest reward.'
        self.rewards = rewards
        self.goals = [
            GoalDeterministic(reward=r) for r in self.rewards
        ]
        self.shuffle_goals_on_reset = shuffle_goals_on_reset
        self.include_reward_permutation = include_reward_permutation

        if seed is None:
            seed = os.getpid() + int(time.time())
            thread_id = threading.current_thread().ident
            if thread_id is not None:
                seed += thread_id
            seed = seed % (2**32 - 1)
            self.np_random, seed = gymnasium.utils.seeding.np_random(seed)

        super().__init__(width=5, height=5, mission_space=default_mission_space)

        assert isinstance(self.observation_space, gymnasium.spaces.Dict)
        if include_reward_permutation:
            self.observation_space = gymnasium.spaces.Dict({
                **self.observation_space.spaces,
                'reward_permutation': gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.rewards),), dtype=np.float32),
            })

    @property
    def reward_permutation(self):
        return [g.reward for g in self.goals]

    def randomize(self):
        self._shuffle_goals()

    def _shuffle_goals(self):
        reward_indices = self.np_random.permutation(len(self.rewards))
        for g,i in zip(self.goals,reward_indices):
            g.reward = self.rewards[i]

    def _gen_grid(self, width, height):
        self.grid = minigrid.core.grid.Grid(width, height)

        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        self.agent_pos = np.array([2, height-2])
        self.agent_dir = self._rand_int(0, 4)

        if self.shuffle_goals_on_reset:
            self._shuffle_goals()

        for i,g in enumerate(self.goals):
            self.put_obj(g, 1+i*2, 1)

    def _reward(self):
        curr_cell = self.grid.get(*self.agent_pos) # type: ignore (where is self.grid assigned?)
        if curr_cell != None and hasattr(curr_cell,'reward'):
            return curr_cell.reward
        breakpoint()
        return 0

    def reset(self):
        obs, info = super().reset()
        if self.include_reward_permutation:
            assert isinstance(obs, dict)
            obs['reward_permutation'] = self.reward_permutation
        if self.shuffle_goals_on_reset:
            self._shuffle_goals()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        info['reward_permutation'] = self.reward_permutation
        if self.include_reward_permutation:
            obs['reward_permutation'] = info['reward_permutation']
        return obs, reward, done, info


register(
    id='MiniGrid-NRoomBanditsSmall-v0',
    entry_point=NRoomBanditsSmall
)


class NRoomBanditsSmallBernoulli(MiniGridEnv):
    def __init__(self, reward_scale=1, prob=0.9, shuffle_goals_on_reset=True, include_reward_permutation=False, seed=None):
        self.mission = 'Reach the goal with the highest reward.'
        self.reward_scale = reward_scale
        self.prob = prob
        self.goals = [
            GoalMultinomial(rewards=[reward_scale,-reward_scale], probs=[prob,1-prob]),
            GoalMultinomial(rewards=[reward_scale,-reward_scale], probs=[1-prob,prob]),
        ]
        self.shuffle_goals_on_reset = shuffle_goals_on_reset
        self.include_reward_permutation = include_reward_permutation

        if seed is None:
            seed = os.getpid() + int(time.time())
            thread_id = threading.current_thread().ident
            if thread_id is not None:
                seed += thread_id
            seed = seed % (2**32 - 1)
            self.np_random, seed = gymnasium.utils.seeding.np_random(seed)

        super().__init__(width=5, height=5, mission_space=default_mission_space)

        assert isinstance(self.observation_space, gymnasium.spaces.Dict)
        if include_reward_permutation:
            self.observation_space = gymnasium.spaces.Dict({
                **self.observation_space.spaces,
                'reward_permutation': gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            })

        # Info
        self._expected_return = 0

    @property
    def reward_permutation(self):
        return [g.expected_value for g in self.goals]

    def randomize(self):
        self._shuffle_goals()

    def _shuffle_goals(self):
        permutation = self.np_random.permutation(2)
        probs = [
                [self.prob, 1-self.prob],
                [1-self.prob, self.prob],
        ]
        for g,i in zip(self.goals,permutation):
            g.probs = probs[i]

    def _gen_grid(self, width, height):
        self.grid = minigrid.core.grid.Grid(width, height)

        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        self.agent_pos = np.array([2, height-2])
        self.agent_dir = self._rand_int(0, 4)

        if self.shuffle_goals_on_reset:
            self._shuffle_goals()

        for i,g in enumerate(self.goals):
            self.put_obj(g, 1+i*2, 1)

    def _reward(self):
        curr_cell = self.grid.get(*self.agent_pos) # type: ignore
        if curr_cell != None and hasattr(curr_cell,'rewards') and hasattr(curr_cell,'probs'):
            return self.np_random.choice(curr_cell.rewards, p=curr_cell.probs)
        breakpoint()
        return 0

    def _expected_reward(self):
        """ Expected reward of the current state """
        curr_cell = self.grid.get(*self.agent_pos) # type: ignore (where is self.grid assigned?)
        if curr_cell is None:
            return 0
        if hasattr(curr_cell,'expected_value'):
            return curr_cell.expected_value
        return 0
    
    @property
    def max_return(self):
        """ Expected return (undiscounted sum) of the optimal policy """
        return max(self.reward_permutation)

    def reset(self):
        obs, info = super().reset()
        if self.include_reward_permutation:
            assert isinstance(obs, dict)
            obs['reward_permutation'] = self.reward_permutation
        if self.shuffle_goals_on_reset:
            self._shuffle_goals()

        self._expected_return = 0

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        info['reward_permutation'] = self.reward_permutation
        if self.include_reward_permutation:
            obs['reward_permutation'] = info['reward_permutation']
        self._expected_return += self._expected_reward()
        info['expected_return'] = self._expected_return
        info['max_return'] = self.max_return
        return obs, reward, done, info


register(
    id='MiniGrid-NRoomBanditsSmallBernoulli-v0',
    entry_point=NRoomBanditsSmallBernoulli
)


class BanditsFetch(MiniGridEnv):
    """
    Environment in which the agent has to fetch a random object
    named using English text strings
    """

    def __init__(
        self,
        size=8,
        num_objs=3,
        num_trials=1,
        reward_correct=1,
        reward_incorrect=-1,
        num_obj_types=2,
        num_obj_colors=6,
        unique_objs=False,
        include_reward_permutation=False,
        seed=None,
    ):
        """
        Args:
            size (int): Size of the grid
            num_objs (int): Number of objects in the environment
            num_trials (int): Number of trials to run with the same set of objects and goal
            reward_correct (int): Reward for picking up the correct object
            reward_incorrect (int): Reward for picking up the incorrect object
            num_obj_types (int): Number of possible object types
            num_obj_colors (int): Number of possible object colors
            unique_objs (bool): If True, each object is unique
            include_reward_permutation (bool): If True, include the reward permutation in the observation
        """
        self.numObjs = num_objs

        self.num_trials = num_trials
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.num_obj_types = num_obj_types
        self.num_obj_colors = num_obj_colors
        self.unique_objs = unique_objs
        self.include_reward_permutation = include_reward_permutation

        self.types = ['key', 'ball']
        self.colors = minigrid.core.constants.COLOR_NAMES

        self.types = self.types[:self.num_obj_types]
        self.colors = self.colors[:self.num_obj_colors]

        if seed is None:
            seed = os.getpid() + int(time.time())
            thread_id = threading.current_thread().ident
            if thread_id is not None:
                seed += thread_id
            seed = seed % (2**32 - 1)
            self.np_random, seed = gymnasium.utils.seeding.np_random(seed)

        super().__init__(
            grid_size=size,
            max_steps=5*size**2*num_trials,
            # Set this to True for maximum speed
            see_through_walls=True,
            mission_space=default_mission_space,
        )

        self.trial_count = 0
        self.objects = []

        assert isinstance(self.observation_space, gymnasium.spaces.Dict)
        if include_reward_permutation:
            self.observation_space = gymnasium.spaces.Dict({
                **self.observation_space.spaces,
                'reward_permutation': gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.types)*len(self.colors),), dtype=np.float32),
            })

    def _gen_grid(self, width, height):
        self.grid = minigrid.core.grid.Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)

        types = self.types
        colors = self.colors

        type_color_pairs = [(t,c) for t in types for c in colors]

        objs = []

        # For each object to be generated
        while len(objs) < self.numObjs:
            objType, objColor = self._rand_elem(type_color_pairs)
            if self.unique_objs:
                type_color_pairs.remove((objType, objColor))

            if objType == 'key':
                obj = minigrid.core.world_object.Key(objColor)
            elif objType == 'ball':
                obj = minigrid.core.world_object.Ball(objColor)
            else:
                raise ValueError(f'Unknown object type: {objType}')

            self.place_obj(obj)
            objs.append(obj)

        self.objects = objs

        # Randomize the player start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        target = objs[self._rand_int(0, len(objs))]
        self.target_type = target.type
        self.target_color = target.color

        descStr = '%s %s' % (self.target_color, self.target_type)

        # Generate the mission string
        idx = self._rand_int(0, 5)
        if idx == 0:
            self.mission = 'get a %s' % descStr
        elif idx == 1:
            self.mission = 'go get a %s' % descStr
        elif idx == 2:
            self.mission = 'fetch a %s' % descStr
        elif idx == 3:
            self.mission = 'go fetch a %s' % descStr
        elif idx == 4:
            self.mission = 'you must fetch a %s' % descStr
        assert hasattr(self, 'mission')

    def reset(self):
        obs, info = super().reset()
        if self.include_reward_permutation:
            assert isinstance(obs, dict)
            obs['reward_permutation'] = self.reward_permutation
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated

        if self.carrying:
            if self.carrying.color == self.target_color and \
               self.carrying.type == self.target_type:
                reward = self.reward_correct
            else:
                reward = self.reward_incorrect
            self.place_obj(self.carrying)
            self.carrying = None
            self.trial_count += 1
            if self.trial_count >= self.num_trials:
                done = True
                self.trial_count = 0

        if self.include_reward_permutation:
            assert isinstance(obs, dict)
            obs['reward_permutation'] = self.reward_permutation

        return obs, reward, done, info

    @property
    def reward_permutation(self):
        r = [self.reward_incorrect, self.reward_correct]
        return [
            r[t == self.target_type and c == self.target_color]
            for t in self.types for c in self.colors
        ]


register(
    id='MiniGrid-BanditsFetch-v0',
    entry_point=BanditsFetch,
)


class MultiRoomEnv(MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        min_num_rooms,
        max_num_rooms,
        max_room_size=10,
        num_trials=100,
        seed = None,
    ):
        assert min_num_rooms > 0
        assert max_num_rooms >= min_num_rooms
        assert max_room_size >= 4

        self.minNumRooms = min_num_rooms
        self.maxNumRooms = max_num_rooms
        self.maxRoomSize = max_room_size

        self.num_trials = num_trials

        self.rooms = []

        if seed is None:
            seed = os.getpid() + int(time.time())
            thread_id = threading.current_thread().ident
            if thread_id is not None:
                seed += thread_id
            seed = seed % (2**32 - 1)
            self.np_random, seed = gymnasium.utils.seeding.np_random(seed)

        super(MultiRoomEnv, self).__init__(
            grid_size=25,
            max_steps=self.maxNumRooms * 20 * self.num_trials,
            mission_space=default_mission_space,
        )

    def _gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = minigrid.core.grid.Grid(width, height)
        wall = minigrid.core.world_object.Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(minigrid.core.constants.COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = minigrid.core.world_object.Door(doorColor)
                self.grid.set(*room.entryDoorPos, entryDoor) # type: ignore
                prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal = minigrid.core.world_object.Goal()
        self.goal_pos = self.place_obj(self.goal, roomList[-1].top, roomList[-1].size)

        self.mission = 'traverse the rooms to get to the goal'

    def _placeRoom(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz+1)
        sizeY = self._rand_int(minSz, maxSz+1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height: # type: ignore
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(gym_minigrid.envs.multiroom.Room( # type: ignore
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for _ in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

    def reset(self):
        obs, info = super().reset()
        self.trial_count = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated

        if done:
            self.trial_count += 1
            if reward > 0:
                reward = 1
            self.grid.set(self.goal_pos[0], self.goal_pos[1], None)
            r = self._rand_int(0, len(self.rooms) - 1)
            self.goal_pos = self.place_obj(self.goal, self.rooms[r].top, self.rooms[r].size)
            if self.trial_count >= self.num_trials:
                done = True
                self.trial_count = 0
            else:
                done = False
            if self.step_count >= self.max_steps:
                done = True

        return obs, reward, done, info


register(
    id='MiniGrid-MultiRoom-v0',
    entry_point=MultiRoomEnv,
)


class Room:
    __slots__ = ['top', 'bottom', 'left', 'right']

    def __init__(self,
            top: int,
            bottom: int,
            left: int,
            right: int,
    ):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __repr__(self):
        return 'Room(top={}, bottom={}, left={}, right={})'.format(
            self.top,
            self.bottom,
            self.left,
            self.right,
        )

    @property
    def width(self):
        return self.right - self.left + 1

    @property
    def height(self):
        return self.bottom - self.top + 1


def room_is_valid(rooms, room, width, height):
    if room.left < 0 or room.right >= width or room.top < 0 or room.bottom >= height:
        return False
    for r in rooms:
        if room.top >= r.bottom:
            continue
        if room.bottom <= r.top:
            continue
        if room.left >= r.right:
            continue
        if room.right <= r.left:
            continue
        return False
    return True


class MultiRoomEnv_v1(MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        min_num_rooms,
        max_num_rooms,
        min_room_size=5,
        max_room_size=10,
        door_prob=0.5,
        num_trials=100,
        fetch_config: dict = None,
        bandits_config: dict = None,
        task_randomization_prob: float = 0,
        max_steps_multiplier: float = 1.,
        shaped_reward_setting: int = None,
        seed = None,
    ):
        """
        Args:
            min_num_rooms (int): minimum number of rooms
            max_num_rooms (int): maximum number of rooms
            min_room_size (int): minimum size of a room. The size includes the walls, so a room of size 3 in one dimension has one occupiable square in that dimension.
            max_room_size (int): maximum size of a room
            door_prob (float): probability of a door being placed in the opening between two rooms.
            num_trials (int): number of trials per episode. A trial ends upon reaching the goal state or picking up an object.
            fetch_config (dict): configuration for the fetch task. If None, no fetch task is used.
            bandits_config (dict): configuration for the bandits task. If None, no bandits task is used.
            task_randomization_prob (float): probability of switching tasks at the beginning of each trial.
            seed (int): random seed.
        """
        assert min_num_rooms > 0
        assert max_num_rooms >= min_num_rooms
        assert max_room_size >= 4

        self.min_num_rooms = min_num_rooms
        self.max_num_rooms = max_num_rooms
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size
        self.door_prob = door_prob

        self.num_trials = num_trials
        self.fetch_config = fetch_config
        self.bandits_config = bandits_config
        self.task_randomization_prob = task_randomization_prob
        self.max_steps_multiplier = max_steps_multiplier
        self.shaped_reward_setting = shaped_reward_setting

        self.rooms = []

        if seed is None:
            seed = os.getpid() + int(time.time())
            thread_id = threading.current_thread().ident
            if thread_id is not None:
                seed += thread_id
            seed = seed % (2**32 - 1)
            self.np_random, seed = gymnasium.utils.seeding.np_random(seed)

        super(MultiRoomEnv_v1, self).__init__(
            grid_size=25,
            max_steps=self.max_num_rooms * 20 * self.num_trials,
            see_through_walls = False,
            mission_space = default_mission_space,
        )

        if shaped_reward_setting is not None:
            assert isinstance(self.observation_space, gymnasium.spaces.Dict)
            self.observation_space = gymnasium.spaces.Dict({
                **self.observation_space.spaces,
                'shaped_reward': gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            })

    def _gen_grid(self, width, height):
        room_list = []
        self.rooms = room_list

        # Choose a random number of rooms to generate
        num_rooms = self._rand_int(self.min_num_rooms, self.max_num_rooms+1)

        # Create first room
        room_height = self._rand_int(self.min_room_size, self.max_room_size+1)
        room_width = self._rand_int(self.min_room_size, self.max_room_size+1)
        top = self._rand_int(1, height - room_height - 1)
        left = self._rand_int(1, width - room_width - 1)
        room_list.append(Room(top, top + room_height - 1, left, left + room_width - 1))

        new_room_openings = [ (0, 'left'), (0, 'right'), (0, 'top'), (0, 'bottom') ]
        while len(room_list) < num_rooms:
            if len(new_room_openings) == 0:
                break

            # Choose a random place to connect the new room to
            r = self._rand_int(0, len(new_room_openings))
            starting_room_index, wall = new_room_openings[r]

            temp_room = self._generate_room(
                    room_list,
                    idx = starting_room_index,
                    wall = wall,
                    min_size = self.min_room_size,
                    max_size = self.max_room_size,
                    width = width,
                    height = height,
            )
            if temp_room is not None:
                room_list.append(temp_room)
                new_room_openings.append((len(room_list)-1, 'left'))
                new_room_openings.append((len(room_list)-1, 'right'))
                new_room_openings.append((len(room_list)-1, 'top'))
                new_room_openings.append((len(room_list)-1, 'bottom'))
            else:
                new_room_openings.remove(new_room_openings[r])

        self.grid = minigrid.core.grid.Grid(width, height)
        self.doors = []
        wall = minigrid.core.world_object.Wall()
        self.wall = wall

        for room in room_list:
            # Look for overlapping walls
            overlapping_walls = {
                'top': [],
                'bottom': [],
                'left': [],
                'right': [],
            }
            for i in range(room.left + 1, room.right):
                if self.grid.get(i,room.top) == wall and self.grid.get(i,room.top+1) is None and self.grid.get(i,room.top-1) is None:
                    overlapping_walls['top'].append((room.top, i))
                if self.grid.get(i,room.bottom) == wall and self.grid.get(i,room.bottom+1) is None and self.grid.get(i,room.bottom-1) is None:
                    overlapping_walls['bottom'].append((room.bottom, i))
            for j in range(room.top + 1, room.bottom):
                if self.grid.get(room.left,j) == wall and self.grid.get(room.left+1,j) is None and self.grid.get(room.left-1,j) is None:
                    overlapping_walls['left'].append((j, room.left))
                if self.grid.get(room.right,j) == wall and self.grid.get(room.right+1,j) is None and self.grid.get(room.right-1,j) is None:
                    overlapping_walls['right'].append((j, room.right))

            # Create room
            # Top wall
            for i in range(room.left, room.right + 1):
                self.grid.set(i, room.top, wall)
            # Bottom wall
            for i in range(room.left, room.right + 1):
                self.grid.set(i, room.bottom, wall)
            # Left wall
            for i in range(room.top, room.bottom + 1):
                self.grid.set(room.left, i, wall)
            # Right wall
            for i in range(room.top, room.bottom + 1):
                self.grid.set(room.right, i, wall)

            # Create doorways between rooms
            for ow in overlapping_walls.values():
                if len(ow) == 0:
                    continue
                opening = self._rand_elem(ow)
                if self.np_random.uniform() > self.door_prob:
                    self.grid.set(opening[1], opening[0], None)
                else:
                    door = minigrid.core.world_object.Door(
                        color = self._rand_elem(minigrid.core.constants.COLOR_NAMES)
                    )
                    self.grid.set(opening[1], opening[0], door)
                    self.doors.append(door)

        self._init_agent()
        self.mission = 'Do whatever'

        # Set max steps
        total_room_sizes = sum([room.height * room.width for room in room_list])
        self.max_steps = int(total_room_sizes * self.num_trials * self.max_steps_multiplier)

    def _init_fetch(self, num_objs, num_obj_types=2, num_obj_colors=6, unique_objs=True, prob=1.0):
        """
        Initialize the fetch task

        Args:
            num_objs: number of objects to generate.
            num_obj_types: number of object types to choose from. Possible object types are "key" and "ball".
            num_obj_colors: number of object colours to choose from. Colours are taken from `gym_minigrid.minigrid.COLOR_NAMES`.
            unique_objs: if True, all objects will be unique. If False, objects can be repeated.
            prob: Probability of obtaining a positive reward upon picking up the target object, or a negative reward upon picking up non-target objects.
        """
        self._fetch_reward_prob = prob

        types = ['key', 'ball'][:num_obj_types]
        colors = minigrid.core.constants.COLOR_NAMES[:num_obj_colors]

        type_color_pairs = [(t,c) for t in types for c in colors]

        objs = []

        # For each object to be generated
        while len(objs) < num_objs:
            obj_type, obj_color = self._rand_elem(type_color_pairs)
            if unique_objs:
                type_color_pairs.remove((obj_type, obj_color))

            if obj_type == 'key':
                obj = minigrid.core.world_object.Key(obj_color)
            elif obj_type == 'ball':
                obj = minigrid.core.world_object.Ball(obj_color)
            else:
                raise ValueError(f'Unknown object type: {obj_type}')

            self.place_obj(obj)
            objs.append(obj)

        self.objects = objs

        # Choose a random object to be picked up
        target = objs[self._rand_int(0, len(objs))]
        self.target_type = target.type
        self.target_color = target.color

    def _init_bandits(self, probs=[1,0]):
        reward_scale = 1
        self.goals = [
            GoalMultinomial(rewards=[reward_scale,-reward_scale], probs=[p,1-p])
            for p in probs
        ]

        for g in self.goals:
            self.place_obj(g, unobstructive=True)

    def _init_agent(self):
        # Randomize the player start position and orientation
        self.agent_pos = self._rand_space()
        self.agent_dir = self._rand_int(0, 4)

    def _rand_space(self):
        """ Find and return the coordinates of a random empty space in the grid """

        room_indices = list(range(len(self.rooms)))
        self.np_random.shuffle(room_indices) # type: ignore

        for r in room_indices:
            # Choose a random room
            room = self.rooms[r]

            # List all spaces in the room
            spaces = [
                (x, y)
                for x in range(room.left+1, room.right)
                for y in range(room.top+1, room.bottom)
            ]
            self.np_random.shuffle(spaces) # type: ignore

            # Choose a random location in the room
            for x, y in spaces:
                # Check if the location is empty
                if self.grid.get(x, y) is not None:
                    continue
                # Check if the agent is here
                if np.array_equal((x,y), self.agent_pos):
                    continue
                return np.array([x, y])

        raise Exception('Could not find a random empty space')

    def _rand_space_unobstructive(self):
        """ Find and return the coordinates of a random empty space in the grid.
        This space is chosen from a set of spaces that would not obstruct access to other parts of the environment if an object were to be placed there.
        """

        room_indices = list(range(len(self.rooms)))
        self.np_random.shuffle(room_indices) # type: ignore

        for r in room_indices:
            # Choose a random room
            room = self.rooms[r]

            # List all spaces in the room
            spaces = [
                (x, y)
                for x in range(room.left+1, room.right)
                for y in range(room.top+1, room.bottom)
            ]
            self.np_random.shuffle(spaces) # type: ignore

            # Choose a random location in the room
            for x, y in spaces:
                # Check if the location is empty
                if self.grid.get(x, y) is not None:
                    continue
                # Check if the agent is here
                if np.array_equal((x,y), self.agent_pos):
                    continue
                # Check if it blocks a doorway
                obstructive = False
                for d in [[0,1],[1,0],[0,-1],[-1,0]]:
                    if self.grid.get(x+d[0], y+d[1]) is self.wall:
                        continue
                    c1 = [d[1]+d[0],d[1]+d[0]]
                    c2 = [-d[1]+d[0],d[1]-d[0]]
                    if self.grid.get(x+c1[0], y+c1[1]) is self.wall and self.grid.get(x+c2[0], y+c2[1]) is self.wall:
                        obstructive = True
                        break

                if obstructive:
                    continue

                return x, y

        raise Exception('Could not find a random empty space')

    def place_obj(self, obj, unobstructive=False):
        if unobstructive:
            pos = self._rand_space_unobstructive()
        else:
            pos = self._rand_space()

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def _generate_room(self, rooms, idx, wall, min_size, max_size, width, height):
        starting_room = rooms[idx]
        new_room = Room(0,0,0,0)
        if wall == 'left' or wall == 'right':
            min_top = max(starting_room.top - max_size + 3, 0)
            max_top = starting_room.bottom - 2
            min_bottom = starting_room.top + 2
            max_bottom = starting_room.bottom + max_size - 3
            if wall == 'left':
                #new_room.right = starting_room.left
                min_right = starting_room.left
                max_right = starting_room.left
                min_left = max(starting_room.left - max_size + 1, 0)
                max_left = starting_room.left - min_size + 1
            else:
                #new_room.left = starting_room.right
                min_left = starting_room.right
                max_left = starting_room.right
                min_right = starting_room.right + min_size - 1
                max_right = starting_room.right + max_size - 1
        else:
            min_left = max(starting_room.left - max_size + 3, 0)
            max_left = starting_room.right - 2
            min_right = starting_room.left + 2
            max_right = starting_room.right + max_size - 3
            if wall == 'top':
                #new_room.bottom = starting_room.top
                min_bottom = starting_room.top
                max_bottom = starting_room.top
                min_top = max(starting_room.top - max_size + 1, 0)
                max_top = starting_room.top - min_size + 1
            else:
                #new_room.top = starting_room.bottom
                min_top = starting_room.bottom
                max_top = starting_room.bottom
                min_bottom = starting_room.bottom + min_size - 1
                max_bottom = starting_room.bottom + max_size - 1
        possible_rooms = [
            (t,b,l,r)
            for t in range(min_top, max_top + 1)
            for b in range(max(min_bottom,t+min_size-1), min(max_bottom + 1, t+max_size))
            for l in range(min_left, max_left + 1)
            for r in range(max(min_right,l+min_size-1), min(max_right + 1, l+max_size))
        ]
        self.np_random.shuffle(possible_rooms) # type: ignore
        for room in possible_rooms:
            new_room.top = room[0]
            new_room.bottom = room[1]
            new_room.left = room[2]
            new_room.right = room[3]
            if room_is_valid(rooms, new_room, width, height):
                return new_room
        return None

    def _randomize_task(self):
        """ Randomize the goal object/states """

        # Fetch task
        target = self.objects[self._rand_int(0, len(self.objects))]
        self.target_type = target.type
        self.target_color = target.color

        # Bandit task
        if self.bandits_config is not None:
            raise NotImplementedError('Task randomization not implemented for bandits')

    def _shaped_reward(self, setting=0):
        """ Return the shaped reward for the current state. """
        if setting == 0: # Distance-based reward bounded by 1
            dest = None
            for obj in self.objects:
                if obj.color != self.target_color:
                    continue
                if obj.type != self.target_type:
                    continue
                dest = obj
            assert dest is not None
            assert dest.cur_pos is not None
            distance = abs(dest.cur_pos[0] - self.agent_pos[0]) + abs(dest.cur_pos[1] - self.agent_pos[1])
            
            reward = 1 / (distance + 1)

            return reward

        raise ValueError(f'Unknown setting: {setting}')

    def reset(self):
        obs, info = super().reset()
        self.trial_count = 0
        if self.fetch_config is not None:
            self._init_fetch(**self.fetch_config)
        if self.bandits_config is not None:
            self._init_bandits(**self.bandits_config)
        if self.shaped_reward_setting is not None:
            obs['shaped_reward'] = self._shaped_reward(self.shaped_reward_setting)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.shaped_reward_setting is not None: # Must happen before `self.carrying` is cleared.
            obs['shaped_reward'] = self._shaped_reward(self.shaped_reward_setting)

        if self.fetch_config is not None:
            if self.carrying:
                reward_correct = self.fetch_config.get('reward_correct', 1)
                reward_incorrect = self.fetch_config.get('reward_incorrect', -1)
                p = self._fetch_reward_prob
                # Check if the agent picked up the correct object
                if self.carrying.color == self.target_color and \
                   self.carrying.type == self.target_type:
                    reward = reward_correct
                    info['regret'] = 0
                else:
                    reward = reward_incorrect
                    info['regret'] = reward_correct*p+reward_incorrect*(1-p) - reward_incorrect*p+reward_correct*(1-p)
                # Flip the reward with some probability
                if self.np_random.uniform() > p:
                    reward *= -1
                # Place the object back in the environment
                self.place_obj(self.carrying)
                # Remove the object from the agent's hand
                self.carrying = None
                # End current trial
                self.trial_count += 1
                # Randomize task if needed
                if self.np_random.uniform() < self.task_randomization_prob:
                    self._randomize_task()

        if self.bandits_config is not None:
            curr_cell = self.grid.get(*self.agent_pos) # type: ignore
            if curr_cell != None and hasattr(curr_cell,'rewards') and hasattr(curr_cell,'probs'):
                # Give a reward
                reward = self.np_random.choice(curr_cell.rewards, p=curr_cell.probs)
                terminated = False
                self.trial_count += 1
                # Teleport the agent to a random location
                self._init_agent()
                # Randomize task if needed
                if self.np_random.uniform() < self.task_randomization_prob:
                    self._randomize_task()

        if self.trial_count >= self.num_trials:
            terminated = True
            self.trial_count = 0

        return obs, reward, terminated, truncated, info


register(
    id='MiniGrid-MultiRoom-v1',
    entry_point=MultiRoomEnv_v1,
)

class MultiRoomBanditsLarge(MultiRoomEnv_v1):
    def __init__(self):
        super().__init__(min_num_rooms=5, max_num_rooms=5, max_room_size=16, fetch_config={'num_objs': 5}, bandits_config={})

register(
    id='MiniGrid-MultiRoom-Large-v1',
    entry_point=MultiRoomBanditsLarge,
)


class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=5,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        num_trials = 100,
        wall = False,
        lava = False,
        seed = None,
    ):
        self.num_trials = num_trials
        self.wall = wall
        self.lava = lava

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        if seed is None:
            seed = os.getpid() + int(time.time())
            thread_id = threading.current_thread().ident
            if thread_id is not None:
                seed += thread_id
            seed = seed % (2**32 - 1)
            self.np_random, seed = gymnasium.utils.seeding.np_random(seed)

        super().__init__(
            grid_size=size,
            max_steps=4*size*size*num_trials,
            # Set this to True for maximum speed
            see_through_walls=True,
            mission_space=default_mission_space,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = minigrid.core.grid.Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create a dividing wall if requested
        if self.wall:
            self.grid.vert_wall(width//2, height//2)

        # Create a lava if requested
        self.lava_obj = minigrid.core.world_object.Lava()
        if self.lava:
            self.place_obj(self.lava_obj)

        # Place a goal square in the bottom-right corner
        self.goal = minigrid.core.world_object.Goal()
        self.place_obj(self.goal)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def reset(self):
        obs, info = super().reset()
        self.trial_count = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated

        curr_cell = self.grid.get(*self.agent_pos) # type: ignore
        if curr_cell is self.goal:
            # Give a reward
            reward = self._reward()
            if self.lava:
                reward = 1 # If we also have a negative reward state, then reaching the goal state will always give a reward of 1 instead of a reward that scales on number of steps taken, since we now care about whether the agent can consistently reach the goal while avoiding bad states rather than how fast it reaches the goal.
            done = False
            self.trial_count += 1
            # Move goal to a random location
            self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
            self.place_obj(self.goal)
        elif curr_cell is self.lava_obj:
            # Give a reward
            reward = -1
            done = False
            self.trial_count += 1
            # Move lava to a random location
            self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
            self.place_obj(self.lava_obj)

        if self.trial_count >= self.num_trials:
            done = True
            self.trial_count = 0

        return obs, reward, done, info


register(
    id='MiniGrid-Empty-Meta-v0',
    entry_point=EmptyEnv,
)


class DelayedRewardEnv(MultiRoomEnv_v1):
    """
    Environment with delayed reward.
    """

    def __init__(self,
        min_num_rooms,
        max_num_rooms,
        min_room_size=5,
        max_room_size=10,
        door_prob=0.5,
        num_trials=100,
        fetch_config={
            'num_objs': 2,
            'num_obj_colors': 6,
            'num_obj_types': 2,
            'prob': 1, # Deterministic rewards
            #'predetermined_objects': [] # Specify the exact objects to use. e.g. `['red ball', 'green key']`
        },
        task_randomization_prob: float = 0,
        max_steps_multiplier: float = 1.,
        shaped_reward_setting: int = None,
        seed = None,
    ):
        """
        Args:
            min_num_rooms (int): minimum number of rooms
            max_num_rooms (int): maximum number of rooms
            min_room_size (int): minimum size of a room. The size includes the walls, so a room of size 3 in one dimension has one occupiable square in that dimension.
            max_room_size (int): maximum size of a room
            door_prob (float): probability of a door being placed in the opening between two rooms.
            num_trials (int): number of trials per episode. A trial ends upon reaching the goal state or picking up an object.
            fetch_config (dict): configuration for the fetch task.
            task_randomization_prob (float): probability of switching tasks at the beginning of each trial.
            seed (int): random seed.
        """
        self.shaped_reward_setting = shaped_reward_setting
        self.fetch_config = fetch_config # Needed to appease pyright
        super().__init__(
            min_num_rooms = min_num_rooms,
            max_num_rooms = max_num_rooms,
            min_room_size = min_room_size,
            max_room_size = max_room_size,
            door_prob = door_prob,
            num_trials = num_trials,
            fetch_config = fetch_config,
            task_randomization_prob = task_randomization_prob,
            max_steps_multiplier = max_steps_multiplier,
            shaped_reward_setting = shaped_reward_setting,
            seed = seed,
        )

    def _gen_grid(self, width, height):
        room_list = []
        self.rooms = room_list

        # Choose a random number of rooms to generate
        num_rooms = self._rand_int(self.min_num_rooms, self.max_num_rooms+1)

        # Create first room
        room_height = self._rand_int(self.min_room_size, self.max_room_size+1)
        room_width = self._rand_int(self.min_room_size, self.max_room_size+1)
        top = self._rand_int(1, height - room_height - 1)
        left = self._rand_int(1, width - room_width - 1)
        room_list.append(Room(top, top + room_height - 1, left, left + room_width - 1))

        new_room_openings = [ (0, 'left'), (0, 'right'), (0, 'top'), (0, 'bottom') ]
        while len(room_list) < num_rooms:
            if len(new_room_openings) == 0:
                break

            # Choose a random place to connect the new room to
            r = self._rand_int(0, len(new_room_openings))
            starting_room_index, wall = new_room_openings[r]

            temp_room = self._generate_room(
                    room_list,
                    idx = starting_room_index,
                    wall = wall,
                    min_size = self.min_room_size,
                    max_size = self.max_room_size,
                    width = width,
                    height = height,
            )
            if temp_room is not None:
                room_list.append(temp_room)
                new_room_openings.append((len(room_list)-1, 'left'))
                new_room_openings.append((len(room_list)-1, 'right'))
                new_room_openings.append((len(room_list)-1, 'top'))
                new_room_openings.append((len(room_list)-1, 'bottom'))
            else:
                new_room_openings.remove(new_room_openings[r])

        self.grid = minigrid.core.grid.Grid(width, height)
        self.doors = []
        wall = minigrid.core.world_object.Wall()
        self.wall = wall

        for room in room_list:
            # Look for overlapping walls
            overlapping_walls = {
                'top': [],
                'bottom': [],
                'left': [],
                'right': [],
            }
            for i in range(room.left + 1, room.right):
                if self.grid.get(i,room.top) == wall and self.grid.get(i,room.top+1) is None and self.grid.get(i,room.top-1) is None:
                    overlapping_walls['top'].append((room.top, i))
                if self.grid.get(i,room.bottom) == wall and self.grid.get(i,room.bottom+1) is None and self.grid.get(i,room.bottom-1) is None:
                    overlapping_walls['bottom'].append((room.bottom, i))
            for j in range(room.top + 1, room.bottom):
                if self.grid.get(room.left,j) == wall and self.grid.get(room.left+1,j) is None and self.grid.get(room.left-1,j) is None:
                    overlapping_walls['left'].append((j, room.left))
                if self.grid.get(room.right,j) == wall and self.grid.get(room.right+1,j) is None and self.grid.get(room.right-1,j) is None:
                    overlapping_walls['right'].append((j, room.right))

            # Create room
            # Top wall
            for i in range(room.left, room.right + 1):
                self.grid.set(i, room.top, wall)
            # Bottom wall
            for i in range(room.left, room.right + 1):
                self.grid.set(i, room.bottom, wall)
            # Left wall
            for i in range(room.top, room.bottom + 1):
                self.grid.set(room.left, i, wall)
            # Right wall
            for i in range(room.top, room.bottom + 1):
                self.grid.set(room.right, i, wall)

            # Create doorways between rooms
            for ow in overlapping_walls.values():
                if len(ow) == 0:
                    continue
                opening = self._rand_elem(ow)
                if self.np_random.uniform() > self.door_prob:
                    self.grid.set(opening[1], opening[0], None)
                else:
                    door = minigrid.core.world_object.Door(
                        color = self._rand_elem(minigrid.core.constants.COLOR_NAMES)
                    )
                    self.grid.set(opening[1], opening[0], door)
                    self.doors.append(door)

        self._init_agent()
        self.mission = 'Pick up the correct object, then reach the green goal square.'

        # Set max steps
        total_room_sizes = sum([room.height * room.width for room in room_list])
        self.max_steps = total_room_sizes * self.num_trials

    def _init_fetch(self,
                    num_objs,
                    num_obj_types=2,
                    num_obj_colors=6,
                    unique_objs=True,
                    prob=1.0,
                    predetermined_objects=[],
        ):
        """
        Initialize the fetch task

        Args:
            num_objs: number of objects to generate.
            num_obj_types: number of object types to choose from. Possible object types are "key" and "ball".
            num_obj_colors: number of object colours to choose from. Colours are taken from `gym_minigrid.minigrid.COLOR_NAMES`.
            unique_objs: if True, all objects will be unique. If False, objects can be repeated.
            prob: Probability of obtaining a positive reward upon picking up the target object, or a negative reward upon picking up non-target objects.
            predetermined_objects: list of objects to be placed in the environment. If there are more objects than `num_objs`, the extra objects will be ignored. If there are fewer, then the rest is filled with randomly generated objects.
                Should be a list of strings of the form "[COLOUR] [OBJECT]", e.g. "red ball".
        """
        self._fetch_reward_prob = prob

        types = ['key', 'ball'][:num_obj_types]
        colors = minigrid.core.constants.COLOR_NAMES[:num_obj_colors]

        type_color_pairs = [(t,c) for t in types for c in colors]

        objs = []

        # For each object to be generated
        predetermined_objects = predetermined_objects[:] # Copy list so we can `pop()`
        while len(objs) < num_objs:
            if len(predetermined_objects) > 0:
                obj_color, obj_type = predetermined_objects.pop().split(' ')
            else:
                obj_type, obj_color = self._rand_elem(type_color_pairs)

            if unique_objs:
                type_color_pairs.remove((obj_type, obj_color))

            if obj_type == 'key':
                obj = minigrid.core.world_object.Key(obj_color)
            elif obj_type == 'ball':
                obj = minigrid.core.world_object.Ball(obj_color)
            else:
                raise ValueError(f'Unknown object type: {obj_type}')

            self.place_obj(obj)
            objs.append(obj)

        self.objects = objs
        self.removed_objects = [] # Objects that were picked up by the agent

        # Choose a random object to be picked up
        target = objs[self._rand_int(0, len(objs))]
        self.target_type = target.type
        self.target_color = target.color
        self.target_object = target # FIXME: This assumes no duplicates

        # Create goal state
        # If the agent enters this state, it will receive a reward based on the objects it has picked up
        self.goal = minigrid.core.world_object.Goal()
        self.place_obj(self.goal, unobstructive=True)

    def _randomize_task(self):
        """ Randomize the goal object/states """
        target = self.objects[self._rand_int(0, len(self.objects))]
        self.target_type = target.type
        self.target_color = target.color

    def _shaped_reward(self, setting=0):
        """ Return the shaped reward for the current state. """
        if setting == 0: # Reward for subtasks
            obj = self.carrying
            if obj is not None:
                if obj.type == self.target_type and obj.color == self.target_color:
                    return 1.0
                else:
                    return -1.0
            curr_cell = self.grid.get(*self.agent_pos) # type: ignore
            if curr_cell == self.goal and len(self.removed_objects) > 0:
                removed_objects_value = self._removed_objects_value()
                if removed_objects_value['target_object_picked_up']:
                    return 1.0
                else:
                    return -1.0
            return 0.0
        elif setting == 1: # Distance-based reward
            # If the agent has picked up the target object, then the destination is the goal state. Otherwise, the destination is the target object.
            dest = self.target_object
            for obj in self.removed_objects:
                if obj.color != self.target_color:
                    continue
                if obj.type != self.target_type:
                    continue
                dest = self.goal
                break
            # Compute the distance to the goal
            assert dest.cur_pos is not None
            distance = abs(dest.cur_pos[0] - self.agent_pos[0]) + abs(dest.cur_pos[1] - self.agent_pos[1])
            # Compute a reward based on the distance
            reward = 1/(distance + 1)

            return reward

        raise ValueError(f'Unknown setting: {setting}')

    def reset(self):
        obs, info = super().reset()
        self.trial_count = 0
        self._init_fetch(**self.fetch_config)
        return obs, info

    def _removed_objects_value(self):
        """ Return the total value of all objects that were picked up. """
        total_regret = 0
        total_expected_value = 0
        total_reward = 0
        target_object_picked_up = False # Whether the target object was picked up

        reward_correct = self.fetch_config.get('reward_correct', 1)
        reward_incorrect = self.fetch_config.get('reward_incorrect', -1)
        p = self._fetch_reward_prob # Probability that the reward is not flipped
        expected_value_correct = reward_correct*p + reward_incorrect*(1-p)
        expected_value_incorrect = reward_incorrect*p + reward_correct*(1-p)
        # Process all objects that were picked up
        for obj in self.removed_objects:
            # Calculate reward and regret for the object
            if obj.color == self.target_color and \
               obj.type == self.target_type:
                reward = reward_correct
                total_expected_value += expected_value_correct
                target_object_picked_up = True
            else:
                reward = reward_incorrect
                total_expected_value += expected_value_incorrect
                total_regret += expected_value_correct - expected_value_incorrect
            # Flip the reward with some probability
            if self.np_random.uniform() > p:
                reward *= -1
            # Sum up reward
            total_reward += reward

        return {
            'total_reward': total_reward,
            'total_regret': total_regret,
            'total_expected_value': total_expected_value,
            'target_object_picked_up': target_object_picked_up,
        }

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        if self.carrying:
            self.removed_objects.append(self.carrying)
            self.carrying = None
        
        curr_cell = self.grid.get(*self.agent_pos) # type: ignore
        total_reward = 0
        total_regret = 0
        if curr_cell == self.goal and len(self.removed_objects) > 0:
            removed_objects_value = self._removed_objects_value()
            reward_correct = self.fetch_config.get('reward_correct', 1)
            reward_incorrect = self.fetch_config.get('reward_incorrect', -1)
            p = self._fetch_reward_prob
            # Process all objects that were picked up
            for obj in self.removed_objects:
                # Calculate reward and regret for the object
                if obj.color == self.target_color and \
                   obj.type == self.target_type:
                    reward = reward_correct
                else:
                    reward = reward_incorrect
                    total_regret += reward_correct*p+reward_incorrect*(1-p) - reward_incorrect*p+reward_correct*(1-p)
                # Flip the reward with some probability
                if self.np_random.uniform() > p:
                    reward *= -1
                # Sum up reward
                total_reward += reward
            # TODO: Replace code above with `removed_objects_value` once it is tested.
            assert total_reward == removed_objects_value['total_reward']
            assert total_regret == removed_objects_value['total_regret']
            info['regret'] = total_regret
            # Place the objects back in the environment
            for obj in self.removed_objects:
                self.place_obj(obj)
            self.removed_objects = []
            # End current trial
            self.trial_count += 1
            # Randomize task if needed
            if self.np_random.uniform() < self.task_randomization_prob:
                self._randomize_task()

        if self.trial_count >= self.num_trials:
            terminated = True
            self.trial_count = 0
        elif self.step_count >= self.max_steps:
            truncated = True
            self.trial_count = 0
        else:
            terminated = False

        return obs, total_reward, terminated, truncated, info


register(
    id='MiniGrid-Delayed-Reward-v0',
    entry_point=DelayedRewardEnv,
)


if __name__ == '__main__':
    env = gym.make('MiniGrid-Delayed-Reward-v0',
            min_num_rooms=4,
            max_num_rooms=6,
            min_room_size=4,
            max_room_size=10,
            fetch_config={'num_objs': 2},
            #bandits_config={
            #    'probs': [0.9, 0.1]
            #},
            #seed=2349918951,
    )
    #env = gym.make('MiniGrid-Empty-Meta-v0',
    #    wall = False,
    #    lava = True,
    #)
    env.reset()
    env.render()
    breakpoint()

