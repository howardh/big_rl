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
                    constant_values=-1, # FIXME: Hack. Change it to 0 when the Gymnasium API allows for different low/high boundaries in the same vectorized environment.
                ),
                high=np.pad(
                    env.action_space.high,
                    tuple((target_shape-action_shape,0) for action_shape, target_shape in zip(env.action_space.low.shape, shape)),
                    mode='constant',
                    #constant_values=0,
                    constant_values=1, # FIXME: Hack. Change it to 0 when the Gymnasium API allows for different low/high boundaries in the same vectorized environment.
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
