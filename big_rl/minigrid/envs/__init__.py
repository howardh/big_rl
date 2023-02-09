import os
from typing_extensions import Literal
from typing import Union, Tuple, List
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


default_mission_space = MissionSpace(mission_func=lambda: "")


def make_env(env_name: str,
        config={},
        minigrid_config={},
        meta_config=None,
        action_shuffle=False) -> gym.Env:
    env = gym.make(env_name, render_mode='rgb_array', **config)
    env = MinigridPreprocessing(env, **minigrid_config)
    if meta_config is not None:
        env = MetaWrapper(env, **meta_config)
    if action_shuffle:
        env = ActionShuffle(env)
    return env


##################################################
# Wrappers
##################################################


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
        if not self._with_mission:
            obs = {k:v for k,v in obs.items() if k != 'mission'}
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # NoopReset
        obs, info = self.env.reset(**kwargs)
        obs = self._resize_obs(obs)
        if not self._with_mission:
            obs = {k:v for k,v in obs.items() if k != 'mission'}
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
            include_reward: bool = True,
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
                ('done', gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)),
                *obs_space,
                ('action', self.env.action_space),
            ]
            if self.include_reward:
                obs_space.append(
                    ('reward', gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32))
                )
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

    def reset(self, seed=None, options=None):
        self.episode_count = 0
        if self.randomize:
            self.env.randomize()
        if self.action_shuffle:
            self._randomize_actions()

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


##################################################
# Utility functions/classes
##################################################


def init_rng(seed=None):
    if seed is None:
        seed = os.getpid() + int(time.time())
        thread_id = threading.current_thread().ident
        if thread_id is not None:
            seed += thread_id
        seed = seed % (2**32 - 1)
    np_random, seed = gymnasium.utils.seeding.np_random(seed)
    return np_random, seed


class PotentialBasedReward:
    def __init__(self, discount, scale=1.0):
        self._scale = scale
        self._discount = discount

        self._prev_potential = None
        self._potential = None

    def update_potential(self, potential):
        self._prev_potential = self._potential
        self._potential = potential

    def reset(self):
        self._prev_potential = None
        self._potential = None

    @property
    def potential(self):
        return self._potential

    @property
    def prev_potential(self):
        return self._prev_potential

    @property
    def reward(self):
        if self.prev_potential is None:
            return 0
        else:
            return (self._discount * self.potential - self.prev_potential) * self._scale


class RewardNoise:
    def __init__(self,
                 noise_type: Literal[None, 'zero', 'gaussian', 'stop'] = None,
                 *args,
                 rng: np.random.RandomState = None
        ):
        self.noise_type = noise_type
        self.args = args

        self.set_rng(rng)
        self.reset()

    def reset(self):
        self._stop_reward: bool = False
        self._step_count: int = 0
        self._trial_count: int = 0
        self._trial_reward: list[float] = [0.]

        self._supervised_steps_in_trial: List[int] = [0]
        self._unsupervised_steps_in_trial: List[int] = [0]

        self.supervised_reward: float = 0.
        """ Total reward obtained while the reward signal is provided """

        self.supervised_trials: int = 0
        """ Total number of trials where the reward signal is provided """

        self.supervised_steps: int = 0
        """ Total number of steps where the reward signal is provided """

        self.unsupervised_reward: float = 0.
        """ Total reward obtained while the reward signal is not provided (e.g. when the signal is cut off due to 'stop' or 'dynamic_stop' noise.) """

        self.unsupervised_trials: int = 0
        """ Total number of trials where the reward signal is not provided (e.g. when the signal is cut off due to 'stop' or 'dynamic_stop' noise.) """

        self.unsupervised_steps: int = 0
        """ Total number of steps where the reward signal is not provided (e.g. when the signal is cut off due to 'stop' or 'dynamic_stop' noise.) """

    def set_rng(self, rng):
        if rng is None:
            self.np_random = np.random
        else:
            self.np_random = rng

    def trial_finished(self):
        self._trial_count += 1
        self._trial_reward.append(0.)

        n_supervised_steps = self._supervised_steps_in_trial[-1]
        n_unsupervised_steps = self._unsupervised_steps_in_trial[-1]

        if n_supervised_steps > 0 and n_unsupervised_steps == 0:
            self.supervised_trials += 1
        elif n_supervised_steps == 0 and n_unsupervised_steps > 0:
            self.unsupervised_trials += 1
        else:
            pass

        self._supervised_steps_in_trial.append(0)
        self._unsupervised_steps_in_trial.append(0)

    def add_noise(self, reward):
        self._step_count += 1

        noise_type = self.noise_type

        if noise_type is None:
            return reward

        if noise_type == 'zero':
            return self._zero_noise(reward, *self.args)
        elif noise_type == 'gaussian':
            return self._gaussian_noise(reward, *self.args)
        elif noise_type == 'stop':
            return self._stop_noise(reward, *self.args)
        elif noise_type == 'dynamic_zero':
            return self._dynamic_zero_noise(reward, *self.args)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    def __call__(self, reward):
        self._trial_reward[-1] += reward
        return self.add_noise(reward)

    def _zero_noise(self, reward: float, zero_when: Union[float, Tuple[int, int]], units: Literal['probability', 'cycle_steps', 'cycle_trials'] = 'probability') -> float:
        """ Set the reward to zero.

        If `units` is 'proability', then `zero_when` is the probability of setting the reward to 0.
        If `units` is 'cycle_steps', then `zero_when` is a tuple `(on_steps, off_steps)`. The reward is unchanged for `on_steps` steps, set to 0 for the next `off_steps` steps, and repeat.
        If `units` is 'cycle_trials', then `zero_when` is a tuple `(on_trials, off_trials)`. The reward is unchanged for `on_trials` trials, set to 0 for the next `off_trials` trials, and repeat.

        Args:
            reward: float, the reward to add noise to
            zero_when: float or tuple of ints, the probability of setting the reward to 0, or the number of steps/trials to set the reward to 0
            units: str, the units of `zero_when`. Can be 'probability', 'cycle_steps', or 'cycle_trials'
        """
        if units == 'probability':
            if self.np_random.uniform() < zero_when:
                self._unsupervised_steps_in_trial[-1] += 1
                self.unsupervised_reward += reward
                return 0
            else:
                self._supervised_steps_in_trial[-1] += 1
                self.supervised_reward += reward
                return reward
        elif units == 'cycle_steps':
            assert len(zero_when) == 2
            period = sum(zero_when)
            assert period > 0

            if (self._step_count-1) % period < zero_when[0]:
                self._supervised_steps_in_trial[-1] += 1
                self.supervised_reward += reward
                return reward
            else:
                self._unsupervised_steps_in_trial[-1] += 1
                self.unsupervised_reward += reward
                return 0
        elif units == 'cycle_trials':
            assert len(zero_when) == 2
            period = sum(zero_when)
            assert period > 0

            if self._trial_count % period < zero_when[0]:
                self._supervised_steps_in_trial[-1] += 1
                self.supervised_reward += reward
                return reward
            else:
                self._unsupervised_steps_in_trial[-1] += 1
                self.unsupervised_reward += reward
                return 0

    def _gaussian_noise(self, reward: float, std: float) -> float:
        """ Add N(0,std) Gaussian noise to the reward """
        return reward + self.np_random.normal(scale=std)

    def _stop_noise(self, reward: float, stop_when: Union[int,float], units: Literal['probability','steps','trials']) -> float:
        """ If `stop_when` is an integer, then stop the reward (i.e. set to 0) after `stop_when` steps. If `stop_when` is a float, then at each step with probability `stop_when`, stop the reward for the rest of the episode. Each time this function is called counts as a step and the step count is reset when `reset()` is called. """

        if units == 'probability':
            if self.np_random.uniform() < stop_when: # stop with a certain probability
                self._stop_reward = True
        elif units == 'steps':
            if self._step_count > stop_when:
                self._stop_reward = True
        elif units == 'trials':
            if self._trial_count >= stop_when:
                self._stop_reward = True

        if self._stop_reward:
            self._unsupervised_steps_in_trial[-1] += 1
            self.unsupervised_reward += reward
            return 0
        else:
            self._supervised_steps_in_trial[-1] += 1
            self.supervised_reward += reward
            return reward

    def _dynamic_zero_noise(self, reward: float, window_size: int, threshold: Tuple[float,float]) -> float:
        """ Dynamically decide when to cut off the reward signal based on performance.

        Args:
            window_size: int, the number of trials to use to compute the average reward
            threshold: tuple of floats, the threshold for stopping the reward. The reward signal is stopped if the average reward is above `max(threshold)`, and resumed if the average reward is below `min(threshold)`.
                The order of the threshold values does not matter.
        """

        stop_threshold = max(threshold) # cut off the reward signal if the average reward is at or above this threshold
        resume_threshold = min(threshold) # resume the reward signal if the average reward is below this threshold
        if self._trial_count < window_size:
            self._supervised_steps_in_trial[-1] += 1
            self.supervised_reward += reward
            return reward

        mean_reward = np.mean(self._trial_reward[-window_size-1:-1])

        if not self._stop_reward and mean_reward >= stop_threshold:
            self._stop_reward = True
        elif self._stop_reward and mean_reward <= resume_threshold:
            self._stop_reward = False

        if self._stop_reward:
            self._unsupervised_steps_in_trial[-1] += 1
            self.unsupervised_reward += reward
            return 0
        else:
            self._supervised_steps_in_trial[-1] += 1
            self.supervised_reward += reward
            return reward


class RewardDelay:
    def __init__(self,
            delay_type: Literal[None, 'fixed', 'random', 'interval'] = None,
            steps: Union[int, Tuple[int,int]] = 0,
            overlap: Literal[None, 'replace', 'sum', 'sum_clipped'] = None,
            rng: np.random.RandomState = None
        ):
        """ Applies a delay to the reward

        Args:
            delay_type: None, 'fixed', 'random', 'interval'
                None: No delay is applied. The reward is returned as is immdiately.
                'fixed': The reward is delayed by a fixed number of steps.
                'random': The reward is delayed by a random number of steps.
                'interval': The reward is given at regular intervals.
            steps:
                If `delay_type` is 'fixed', then this is the number of steps to delay the reward.
                If `delay_type` is 'random', then this is a tuple of the form (min_steps, max_steps) specifying the range of steps to delay the reward. The delay is chosen uniformly at random within this range.
                If `delay_type` is 'interval', then this can be either an integer or a tuple of the form (min_steps, max_steps). If it is an integer, then a reward is given on every `steps` step. If it is a tuple, then the time between each reward is chosen uniformly at random from the range (min_steps, max_steps), and is sampled after each reward.
            overlap: How to handle rewards whose delays overlap, causing them to show up out of order.
                'replace': The most recent reward replaces the previous reward.
                'sum': The rewards are summed.
                'sum_clipped': The rewards are summed, then clipped between -1 and 1.
        """
        self.delay_type = delay_type
        self.steps = steps
        self.overlap = overlap

        if rng is None:
            self.np_random = np.random.RandomState()
        else:
            self.np_random = rng

        self.reset()

    def reset(self):
        if self.delay_type == 'fixed':
            assert type(self.steps) is int
            if self.steps >= 0:
                self._buffer = [0] * (self.steps + 1)
            else:
                self._buffer = [] 
            self._buffer_idx = 0

        if self.delay_type == 'random':
            assert type(self.steps) is tuple
            _, max_steps = self.steps
            self._buffer = [0] * (max_steps + 1)
            self._buffer_idx = 0

        if self.delay_type == 'interval':
            self._buffer = [0]
            self._buffer_idx = 1

            if isinstance(self.steps, int):
                assert self.steps > 0
            elif isinstance(self.steps, tuple):
                min_steps, max_steps = self.steps
                assert min_steps > 0
                assert max_steps > 0
                assert min_steps <= max_steps

    def trial_finished(self):
        ...

    def delay(self, reward: float) -> float:
        if self.delay_type is None:
            return reward

        if self.delay_type == 'fixed':
            return self._fixed_delay(reward, self.steps)
        elif self.delay_type == 'random':
            return self._random_delay(reward, self.steps, self.overlap)
        elif self.delay_type == 'interval':
            return self._interval_delay(reward, self.steps, self.overlap)

    def __call__(self, reward: float) -> float:
        return self.delay(reward)

    def _fixed_delay(self, reward: float, steps: int) -> float:
        """ Delay the reward by `steps` steps. If `steps` is negative, then the reward is delayed indefinitely (i.e. will always be 0). """
        if steps < 0:
            return 0

        self._buffer_idx = (self._buffer_idx + 1) % len(self._buffer)

        delayed_idx = (self._buffer_idx + steps) % len(self._buffer)
        self._buffer[delayed_idx] = reward
        return self._buffer[self._buffer_idx]

    def _random_delay(self, reward: float, steps: Tuple[int,int], overlap: str) -> float:
        """ Delay the reward by a random number of steps in the range `steps`. """
        self._buffer[self._buffer_idx] = 0
        self._buffer_idx = (self._buffer_idx + 1) % len(self._buffer)

        min_steps, max_steps = steps
        delay = self.np_random.randint(min_steps, max_steps + 1)
        delayed_idx = (self._buffer_idx + delay) % len(self._buffer)

        if overlap == 'replace':
            self._buffer[delayed_idx] = reward
        elif overlap == 'sum' or overlap == 'sum_clipped':
            self._buffer[delayed_idx] += reward

        if overlap == 'replace' or overlap == 'sum':
            return self._buffer[self._buffer_idx]
        elif overlap == 'sum_clipped':
            return np.clip(self._buffer[self._buffer_idx], -1, 1)
        else:
            raise ValueError(f'Unknown overlap type: {overlap}')

    def _interval_delay(self, reward: float, steps: Union[int,Tuple[int,int]], overlap: str) -> float:
        """ Give a reward every `steps` steps. If `steps` is a tuple, then the time between each reward is chosen uniformly at random from the range `steps`. """
        self._buffer_idx -= 1

        if type(steps) is int:
            min_steps = steps
            max_steps = steps
        else:
            min_steps, max_steps = steps

        if self._buffer_idx <= 0:
            self._buffer_idx = self.np_random.randint(min_steps, max_steps + 1)
            self._buffer[0] = 0

        if overlap == 'replace':
            self._buffer[0] = reward
        elif overlap == 'sum' or overlap == 'sum_clipped':
            self._buffer[0] += reward

        if self._buffer_idx <= 1:
            if overlap == 'replace' or overlap == 'sum':
                return self._buffer[0]
            elif overlap == 'sum_clipped':
                return np.clip(self._buffer[0], -1, 1)
            else:
                raise ValueError(f'Unknown overlap type: {overlap}')
        else:
            return 0


class RewardDelayedStart:
    def __init__(self,
            delay_type: Literal[None, 'fixed', 'random'] = None,
            start_when: Union[int, Tuple[int,int]] = 0,
            units: Literal['steps', 'trials'] = 'steps',
            rng: np.random.RandomState = None
        ):
        """ Wait a certain number of steps or trials before giving a reward.
        """

        # Validate inputs
        assert units in ['steps', 'trials']
        if delay_type == 'fixed':
            assert isinstance(start_when, int)
        elif delay_type == 'random':
            assert isinstance(start_when, tuple)
            assert len(start_when) == 2

        self._delay_type = delay_type
        self._start_when = start_when
        self._units = units

        if rng is None:
            self.np_random = np.random.RandomState()
        else:
            self.np_random = rng

        self.reset()

    def reset(self):
        self._step_count = 0
        self._trial_count = 0
        if self._delay_type is None:
            self._delay = 0
        elif self._delay_type == 'fixed':
            self._delay = self._start_when
        elif self._delay_type == 'random':
            self._delay = self.np_random.randint(*self._start_when)
        else:
            raise ValueError(f'Unknown delay type: {self._delay_type}')

    def trial_finished(self):
        self._trial_count += 1

    def delay(self, reward: float) -> float:
        self._step_count += 1
        if self._delay_type is None:
            return reward

        if self._units == 'steps':
            if self._step_count > self._delay:
                return reward
            else:
                return 0
        elif self._units == 'trials':
            if self._trial_count >= self._delay:
                return reward
            else:
                return 0
        else:
            raise ValueError(f'Unknown units: {self._units}')

    def __call__(self, reward: float) -> float:
        return self.delay(reward)


##################################################
# Minigrid Objects
##################################################


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


##################################################
# Environments
##################################################


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
        shaped_reward_config: dict = None,
        reward_type: Literal['standard', 'pbrs'] = 'standard',
        pbrs_scale: float = 1.,
        pbrs_discount: float = 0.99,
        seed = None,
        render_mode = 'rgb_array',
    ):
        """
        Args:
            min_num_rooms (int): minimum number of rooms
            max_num_rooms (int): maximum number of rooms
            min_room_size (int): minimum size of a room. The size includes the walls, so a room of size 3 in one dimension has one occupiable square in that dimension.
            max_room_size (int): maximum size of a room
            door_prob (float): probability of a door being placed in the opening between two rooms.
            num_trials (int): number of trials per episode. A trial ends upon reaching the goal state or picking up an object. If set to -1, the environment will run indefinitely.
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
        self.shaped_reward_config = shaped_reward_config
        if shaped_reward_config is not None:
            self.shaped_reward_noise = RewardNoise(
                    *shaped_reward_config.get('noise', tuple()))
            self.shaped_reward_delay = RewardDelay(
                    *shaped_reward_config.get('delay', tuple()))
            self.shaped_reward_delayed_start = RewardDelayedStart(
                    *shaped_reward_config.get('delayed_start', tuple()))
            self.shaped_reward_transforms = [
                self.shaped_reward_delay,
                self.shaped_reward_noise,
                self.shaped_reward_delayed_start,
            ]
        else:
            self.shaped_reward_transforms = []
        self.reward_type = reward_type

        self.pbrs = PotentialBasedReward(
            discount=pbrs_discount,
            scale=pbrs_scale,
        )

        self.rooms = []

        self.np_random, seed = init_rng(seed)

        super(MultiRoomEnv_v1, self).__init__(
            grid_size=25,
            max_steps=self.max_num_rooms * 20 * self.num_trials,
            see_through_walls = False,
            mission_space = MissionSpace(mission_func = lambda: 'Do whatever'),
            render_mode = render_mode,
        )

        assert isinstance(self.observation_space, gymnasium.spaces.Dict)
        if shaped_reward_config is not None:
            self.observation_space = gymnasium.spaces.Dict({
                **self.observation_space.spaces,
                'shaped_reward': gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            })

        self._agent_pos_prev = None

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

        assert isinstance(self.observation_space, gymnasium.spaces.Dict)
        self.mission = self.observation_space.spaces['mission'].sample()

        # Set max steps
        total_room_sizes = sum([room.height * room.width for room in room_list])
        self.max_steps = int(total_room_sizes * self.num_trials * self.max_steps_multiplier)
        if self.max_steps < 0:
            self.max_steps = float('inf')

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

    def _get_dest(self):
        dest = None
        for obj in self.objects:
            if obj.color != self.target_color:
                continue
            if obj.type != self.target_type:
                continue
            dest = obj
        assert dest is not None
        assert dest.cur_pos is not None
        return dest

    def _shaped_reward(self, config: dict):
        """ Return the shaped reward for the current state.

        Args:
            config: dictionary containing the following keys:
                - 'reward_type': 'zero', 'inverse distance', 'adjacent to subtask'
                - 'noise': Tuple of (noise type, *noise params). Possible noise types with their parameters:
                    - ('zero', p): Shaped reward is 0 with probability p
                    - ('gaussian', std): Zero-mean Gaussian noise with standard deviation std is added to the shaped reward
                    - ('stop', n): If n>1, the shaped reward is set to 0 after n steps. If 0<n<1, the shaped reward will be set to 0 for the rest of the episode with probability n at each time step.
        """
        reward_type = config['type'].lower()

        def compute_reward():
            all_dests = self.objects # List of all possible destinations

            if reward_type == 'subtask': # Reward for subtasks. There is only one task here, so this is just the regular reward signal.
                obj = self.carrying
                if obj is not None:
                    if obj.type == self.target_type and obj.color == self.target_color:
                        return 1.0
                    else:
                        return -1.0
                return 0.0

            elif reward_type == 'inverse distance': # Distance-based reward bounded by 1
                dest = self._get_dest()
                distance = abs(dest.cur_pos[0] - self.agent_pos[0]) + abs(dest.cur_pos[1] - self.agent_pos[1])
                
                reward = 1 / (distance + 1)

                return reward

            elif reward_type == 'zero': # Always 0. Use to test the agent's ability to handle having a shaped reward signal but the signal is constant.
                return 0

            elif reward_type == 'pbrs': # Same as inverse distance, but converted to a potential-based shaped reward
                raise NotImplementedError('Potential-based shaped reward not implemented')

            elif reward_type == 'adjacent to subtask':
                dest = self._get_dest()

                assert self.agent_pos is not None
                if self._agent_pos_prev is None:
                    return 0

                dist = abs(dest.cur_pos[0] - self.agent_pos[0]) + abs(dest.cur_pos[1] - self.agent_pos[1])
                dist_prev = abs(dest.cur_pos[0] - self._agent_pos_prev[0]) + abs(dest.cur_pos[1] - self._agent_pos_prev[1])
                all_dists = [abs(obj.cur_pos[0] - self.agent_pos[0]) + abs(obj.cur_pos[1] - self.agent_pos[1]) for obj in all_dests]
                all_dists_prev = [abs(obj.cur_pos[0] - self._agent_pos_prev[0]) + abs(obj.cur_pos[1] - self._agent_pos_prev[1]) for obj in all_dests]

                if dist == 1:
                    if dist_prev != 1:
                        return 1
                    else:
                        return 0
                elif min(all_dists) == 1 and min(all_dists_prev) != 1:
                    return -1
                else:
                    return 0

            raise ValueError(f'Unknown reward type: {reward_type}')

        reward = compute_reward()
        for transform in self.shaped_reward_transforms:
            reward = transform(reward)
        #reward = self.shaped_reward_delay(reward)
        #reward = self.shaped_reward_noise(reward)
        return reward

    def _potential(self):
        """ Potential function used for potential-based shaped reward. """
        dest = self._get_dest()
        dist = abs(dest.cur_pos[0] - self.agent_pos[0]) + abs(dest.cur_pos[1] - self.agent_pos[1])
        max_size = self.width + self.height
        return -dist / max_size

    @property
    def goal_str(self):
        return f'{self.target_color} {self.target_type}'

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.trial_count = 0
        self._agent_pos_prev = None
        if self.fetch_config is not None:
            self._init_fetch(**self.fetch_config)
        if self.bandits_config is not None:
            self._init_bandits(**self.bandits_config)
        if self.shaped_reward_config is not None:
            obs['shaped_reward'] = np.array([self._shaped_reward(self.shaped_reward_config)], dtype=np.float32)
            #self.shaped_reward_noise.reset()
            info['supervised_reward'] = self.shaped_reward_noise.supervised_reward
            info['unsupervised_reward'] = self.shaped_reward_noise.unsupervised_reward
            info['supervised_trials'] = self.shaped_reward_noise.supervised_trials
            info['unsupervised_trials'] = self.shaped_reward_noise.unsupervised_trials
            for transform in self.shaped_reward_transforms:
                transform.reset()
        self.pbrs.reset()
        info['reward'] = 0
        return obs, info

    def step(self, action):
        self._agent_pos_prev = self.agent_pos

        obs, reward, terminated, truncated, info = super().step(action)

        if self.shaped_reward_config is not None: # Must happen before `self.carrying` is cleared.
            obs['shaped_reward'] = np.array([self._shaped_reward(self.shaped_reward_config)], dtype=np.float32)
            info['supervised_reward'] = self.shaped_reward_noise.supervised_reward
            info['unsupervised_reward'] = self.shaped_reward_noise.unsupervised_reward
            info['supervised_trials'] = self.shaped_reward_noise.supervised_trials
            info['unsupervised_trials'] = self.shaped_reward_noise.unsupervised_trials

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
                #if self.shaped_reward_config is not None:
                #    self.shaped_reward_noise.trial_finished()
                for transform in self.shaped_reward_transforms:
                    transform.trial_finished()
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
                #if self.shaped_reward_config is not None:
                #    self.shaped_reward_noise.trial_finished()
                for transform in self.shaped_reward_transforms:
                    transform.trial_finished()
                # Teleport the agent to a random location
                self._init_agent()
                # Randomize task if needed
                if self.np_random.uniform() < self.task_randomization_prob:
                    self._randomize_task()

        if self.num_trials > 0 and self.trial_count >= self.num_trials:
            terminated = True
            self.trial_count = 0

        info['reward'] = reward
        if self.reward_type == 'pbrs':
            self.pbrs.update_potential(self._potential())
            return obs, reward+self.pbrs.reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info


register(
    id='MiniGrid-MultiRoom-v1',
    entry_point=MultiRoomEnv_v1,
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
        shaped_reward_config: dict = None,
        reward_type: Literal['standard', 'pbrs'] = 'standard',
        seed = None,
        render_mode = 'rgb_array',
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
            reward_type (str): type of reward to use. 'standard' gives a reward only after an object is picked up and the agent steps on the green square. `pbrs` is the potential-based shaped reward. The potential function used is the negative of the Manhattan distance to the current subtask, normalized to be bounded between [-1,0].
            seed (int): random seed.
        """
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
            shaped_reward_config = shaped_reward_config,
            reward_type = reward_type,
            seed = seed,
            render_mode = render_mode,
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

        assert isinstance(self.observation_space, gymnasium.spaces.Dict)
        self.mission = self.observation_space.spaces['mission'].sample()

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

    def _shaped_reward(self, config: dict):
        """ Return the shaped reward for the current state. """
        reward_type = config['type'].lower()

        def get_dest():
            for obj in self.removed_objects:
                if obj.color != self.target_color:
                    continue
                if obj.type != self.target_type:
                    continue
                assert self.goal is not None
                assert self.goal.cur_pos is not None
                return self.goal
            assert self.target_object is not None
            assert self.target_object.cur_pos is not None
            return self.target_object

        def compute_reward():
            all_dests = [self.goal, *self.objects]

            if reward_type == 'subtask': # Reward for subtasks
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

            elif reward_type == 'inverse distance': # Distance-based reward
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

            elif reward_type == 'adjacent to subtask':
                dest = get_dest()

                assert dest.cur_pos is not None
                assert self.agent_pos is not None
                if self._agent_pos_prev is None:
                    return 0.0

                dist = abs(dest.cur_pos[0] - self.agent_pos[0]) + abs(dest.cur_pos[1] - self.agent_pos[1])
                dist_prev = abs(dest.cur_pos[0] - self._agent_pos_prev[0]) + abs(dest.cur_pos[1] - self._agent_pos_prev[1])
                all_dists = [abs(dest.cur_pos[0] - obj.cur_pos[0]) + abs(dest.cur_pos[1] - obj.cur_pos[1]) for obj in all_dests]

                if dist == 1:
                    if dist_prev != 1:
                        return 1
                    else:
                        return 0
                elif min(all_dists) == 1:
                    return -1
                else:
                    return 0

            raise ValueError(f'Unknown reward type: {reward_type}')

        reward = compute_reward()
        for transform in self.shaped_reward_transforms:
            reward = transform(reward)
        #reward = self.shaped_reward_delay(reward)
        #reward = self.shaped_reward_noise(reward)
        return reward

    def reset(self, seed=None, options=None):
        obs, info = MiniGridEnv.reset(self, seed=seed, options=options)
        self.trial_count = 0
        self._agent_pos_prev = None
        self._init_fetch(**self.fetch_config)
        if self.shaped_reward_config is not None:
            obs['shaped_reward'] = np.array([self._shaped_reward(self.shaped_reward_config)], dtype=np.float32)
            info['supervised_reward'] = self.shaped_reward_noise.supervised_reward
            info['unsupervised_reward'] = self.shaped_reward_noise.unsupervised_reward
            info['supervised_trials'] = self.shaped_reward_noise.supervised_trials
            info['unsupervised_trials'] = self.shaped_reward_noise.unsupervised_trials
        return obs, info

    def _removed_objects_value(self):
        """ Return the total value of all objects that were picked up. """
        total_regret: float = 0
        total_expected_value: float = 0
        total_reward: float = 0
        target_object_picked_up = False # Whether the target object was picked up

        reward_correct: float = self.fetch_config.get('reward_correct', 1)
        reward_incorrect: float = self.fetch_config.get('reward_incorrect', -1)
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
        self._agent_pos_prev = self.agent_pos

        obs, _, terminated, truncated, info = MiniGridEnv.step(self, action)

        if self.shaped_reward_config is not None: # Must happen before `self.carrying` is cleared.
            obs['shaped_reward'] = np.array([self._shaped_reward(self.shaped_reward_config)], dtype=np.float32)
            info['supervised_reward'] = self.shaped_reward_noise.supervised_reward
            info['unsupervised_reward'] = self.shaped_reward_noise.unsupervised_reward
            info['supervised_trials'] = self.shaped_reward_noise.supervised_trials
            info['unsupervised_trials'] = self.shaped_reward_noise.unsupervised_trials

        if self.carrying:
            self.removed_objects.append(self.carrying)
            self.carrying = None
        
        curr_cell = self.grid.get(*self.agent_pos) # type: ignore
        total_reward = 0
        total_regret = 0
        if curr_cell == self.goal and len(self.removed_objects) > 0:
            removed_objects_value = self._removed_objects_value()
            reward_correct: float = self.fetch_config.get('reward_correct', 1)
            reward_incorrect: float = self.fetch_config.get('reward_incorrect', -1)
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
            #if self.shaped_reward_config is not None:
            #    self.shaped_reward_noise.trial_finished()
            for transform in self.shaped_reward_transforms:
                transform.trial_finished()
            # Randomize task if needed
            if self.np_random.uniform() < self.task_randomization_prob:
                self._randomize_task()

        if self.num_trials > 0 and self.trial_count >= self.num_trials:
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

