import gymnasium
import numpy as np


class PadObservation(gymnasium.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)

        if not isinstance(env.observation_space, gymnasium.spaces.Box):
            raise TypeError(f'Expected observation space to be of type Box, got {type(env.observation_space)}')
        if len(env.observation_space.shape) != len(shape):
            raise ValueError(f'Expected observation space and padded observation space to have the same number of dimensions. Received sizes {len(env.observation_space.shape)} and {len(shape)}.')

        self.shape = shape
        self.observation_space = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.pad(
                observation,
                tuple((0,target_shape-obs_shape) for obs_shape, target_shape in zip(observation.shape, self.shape)),
                mode='constant',
                constant_values=0,
        )


class PadAction(gymnasium.ActionWrapper):
    def __init__(self, env, shape):
        super().__init__(env)

        if not isinstance(env.action_space, gymnasium.spaces.Box):
            raise TypeError(f'Expected action space to be of type Box, got {type(env.action_space)}')
        if len(env.action_space.shape) != len(shape):
            raise ValueError(f'Expected action space and padded action space to have the same number of dimensions. Received sizes {len(env.action_space.shape)} and {len(shape)}.')

        self.shape = shape
        self.action_space = gymnasium.spaces.Box(
                low=np.pad(
                    env.action_space.low, 
                    tuple((target_shape-action_shape,0) for action_shape, target_shape in zip(env.action_space.low.shape, shape)),
                    mode='constant',
                    #constant_values=0,
                    constant_values=-1, # FIXME: Hack. Change it to 0 when the Gymnasium API allows for different low/high boundaries in the same vectorized environment. This only works for now because the envirionments I'm using all have the same low/high values.
                ),
                high=np.pad(
                    env.action_space.high,
                    tuple((target_shape-action_shape,0) for action_shape, target_shape in zip(env.action_space.low.shape, shape)),
                    mode='constant',
                    #constant_values=0,
                    constant_values=1, # FIXME: Hack. Change it to 0 when the Gymnasium API allows for different low/high boundaries in the same vectorized environment. This only works for now because the envirionments I'm using all have the same low/high values.
                ),
                shape=shape,
                dtype=np.float32,
        )

        self.original_shape = env.action_space.shape

    def action(self, action: np.ndarray) -> np.ndarray:
        indices = [
            slice(0, s) for s in self.original_shape
        ]
        return action.__getitem__(*indices)


class ToDictObservation(gymnasium.ObservationWrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self.key = key

        self.observation_space = gymnasium.spaces.Dict({key: env.observation_space})

    def observation(self, observation: np.ndarray) -> dict:
        return {self.key: observation}


class AddDummyObservation(gymnasium.ObservationWrapper):
    def __init__(self, env, key, value):
        super().__init__(env)

        self.key = key
        self.value = value

        if not isinstance(env.observation_space, gymnasium.spaces.Dict):
            raise TypeError(f'AddDummyObservation can only be applied to environments with a Dict observation space. Received {type(env.observation_space)}')
        if key in env.observation_space.spaces:
            raise ValueError(f'Key {key} already exists in observation space {env.observation_space}')

        self.observation_space = gymnasium.spaces.Dict({
            **env.observation_space.spaces,
            key: gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=[1], dtype=np.float32),
        })

    def observation(self, observation: dict) -> dict:
        observation[self.key] = [self.value]
        return observation


class AddDummyInfo(gymnasium.Wrapper):
    def __init__(self, env, key, value, overwrite=False):
        super().__init__(env)

        self.key = key
        self.value = value
        self.overwrite = overwrite

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        if self.overwrite or self.key not in info:
            info[self.key] = self.value
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action) # type: ignore
        if self.overwrite or self.key not in info:
            info[self.key] = self.value
        return observation, reward, terminated, truncated, info


class ShuffleObservation(gymnasium.Wrapper):
    def __init__(self, env, key=None):
        super().__init__(env)

        self.key = key

        if key is None:
            if not isinstance(env.observation_space, gymnasium.spaces.Box):
                raise TypeError(f'Attempted to shuffle a non-Box observation space. Received {type(env.observation_space)}')
            if not len(env.observation_space.shape) == 1:
                raise ValueError(f'Only 1D vector observations can be shuffled. Received observation shape {env.observation_space.shape}')
        else:
            if not isinstance(env.observation_space, gymnasium.spaces.Dict):
                raise TypeError(f'A key was provided, but the observation space is not a Dict. Received {type(env.observation_space)}')
            if key not in env.observation_space.spaces:
                raise ValueError(f'The provided key was not found in the observation space. Received {key}. Available keys are {env.observation_space.spaces.keys()}')
            if not isinstance(env.observation_space.spaces[key], gymnasium.spaces.Box):
                raise TypeError(f'Attempted to shuffle a non-Box observation space. Received {type(env.observation_space.spaces[key])}')

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)

        if self.key is None:
            assert isinstance(obs, np.ndarray)
            self.permutation = np.random.permutation(np.arange(obs.shape[0]))
        else:
            assert isinstance(obs, dict)
            assert isinstance(obs[self.key], np.ndarray)
            self.permutation = np.random.permutation(np.arange(obs[self.key].shape[0]))

        return self.shuffle_observation(obs), info # type: ignore

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action) # type: ignore
        return self.shuffle_observation(obs), reward, terminated, truncated, info # type: ignore

    def shuffle_observation(self, observation: np.ndarray | dict) -> np.ndarray | dict:
        if self.key is None:
            assert isinstance(observation, np.ndarray)
            return observation[self.permutation]
        else:
            assert isinstance(observation, dict)
            observation[self.key] = observation[self.key][self.permutation]
            return observation


class ShuffleAction(gymnasium.Wrapper):
    # XXX: Untested
    def __init__(self, env, key=None):
        super().__init__(env)

        self.key = key

        if key is None:
            if not isinstance(env.action_space, gymnasium.spaces.Box):
                raise TypeError(f'Attempted to shuffle a non-Box action space. Received {type(env.action_space)}')
            if not len(env.action_space.shape) == 1:
                raise ValueError(f'Only 1D vector actions can be shuffled. Received action shape {env.action_space.shape}')
        else:
            if not isinstance(env.action_space, gymnasium.spaces.Dict):
                raise TypeError(f'A key was provided, but the action space is not a Dict. Received {type(env.action_space)}')
            if key not in env.action_space.spaces:
                raise ValueError(f'The provided key was not found in the action space. Received {key}. Available keys are {env.action_space.spaces.keys()}')
            if not isinstance(env.action_space.spaces[key], gymnasium.spaces.Box):
                raise TypeError(f'Attempted to shuffle a non-Box action space. Received {type(env.action_space.spaces[key])}')
        # TODO: Implement for Discrete action spaces

    def reset(self, *args, **kwargs):
        if self.key is None:
            shuffled_action_space = self.action_space
        else:
            assert isinstance(self.action_space, gymnasium.spaces.Dict)
            shuffled_action_space = self.action_space[self.key]
        assert shuffled_action_space.shape is not None
        self.permutation = np.random.permutation(np.arange(shuffled_action_space.shape[0]))

        return self.env.reset(*args, **kwargs)

    def step(self, action):
        action = self.shuffle_action(action) # type: ignore
        obs, reward, terminated, truncated, info = self.env.step(action) # type: ignore
        return obs, reward, terminated, truncated, info # type: ignore

    def shuffle_action(self, action: np.ndarray | dict) -> np.ndarray | dict:
        if self.key is None:
            assert isinstance(action, np.ndarray)
            return action[self.permutation]
        else:
            assert isinstance(action, dict)
            action[self.key] = action[self.key][self.permutation]
            return action


def _make_mask(space, p, ignore_unsupported=False):
    if isinstance(space, gymnasium.spaces.Box):
        obs_shape = space.shape
        assert obs_shape is not None
        obs_size = np.prod(obs_shape)
        num_non_visible = int(p * obs_size)
        num_visible = obs_size - num_non_visible
        visible = np.array([True] * num_visible + [False] * num_non_visible)
        np.random.shuffle(visible)
        mask = visible.reshape(obs_shape)
        return mask
    if isinstance(space, gymnasium.spaces.Dict):
        return {
            key: _make_mask(subspace, p)
            for key, subspace in space.spaces.items()
        }
    if isinstance(space, gymnasium.spaces.Tuple):
        return tuple(
            _make_mask(sub_space, p)
            for sub_space in space.spaces
        )
    if ignore_unsupported:
        return None
    raise TypeError(f'Unsupported space type {type(space)}')


def _apply_mask(x, mask, value, space):
    if mask is None:
        return x

    if isinstance(space, gymnasium.spaces.Box):
        if isinstance(x, list) or isinstance(x, tuple):
            x = np.array(x)
        return np.where(mask, x, value)
    if isinstance(space, gymnasium.spaces.Dict):
        assert isinstance(x, dict)
        return {
            key: _apply_mask(sub_x, mask[key], value, space.spaces[key])
            for (key, sub_x) in x.items()
        }
    if isinstance(space, gymnasium.spaces.Tuple):
        if isinstance(x, tuple):
            return tuple(
                _apply_mask(sub_x, sub_mask, value, sub_space)
                for (sub_x, sub_mask, sub_space) in zip(x, mask, space.spaces)
            )
        if isinstance(x, list):
            return [
                _apply_mask(sub_x, sub_mask, value, sub_space)
                for (sub_x, sub_mask, sub_space) in zip(x, mask, space.spaces)
            ]
    raise TypeError(f'Unsupported type {type(x)}')


class OccludeObservation(gymnasium.Wrapper):
    def __init__(self, env, p: float, value: float = 0.0):
        super().__init__(env)

        self.p = p
        self.value = value

        ## Create a mask now to make sure it's work. Fail fast and give a good error message.
        #self.mask = _make_mask(env.observation_space, p)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.reset_mask()
        return self.mask_observation(obs), info # type: ignore

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action) # type: ignore
        return self.mask_observation(obs), reward, terminated, truncated, info # type: ignore

    def reset_mask(self):
        #obs_shape = self.observation_space.shape
        #assert obs_shape is not None
        #obs_size = np.prod(obs_shape)
        #num_non_visible = int(self.p * obs_size)
        #num_visible = obs_size - num_non_visible
        #visible = np.array([True] * num_visible + [False] * num_non_visible)
        #np.random.shuffle(visible)

        #self.mask = visible.reshape(obs_shape)
        self.mask = _make_mask(self.observation_space, self.p, ignore_unsupported=True)

    def mask_observation(self, observation: np.ndarray):
        #assert isinstance(observation, np.ndarray)
        #return np.where(self.mask, observation, self.value)
        return _apply_mask(observation, self.mask, self.value, self.observation_space)
