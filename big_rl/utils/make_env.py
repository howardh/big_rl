import copy
from typing import NamedTuple, Any
from enum import Enum
import itertools
import yaml

import gymnasium
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStack, TimeLimit
from gymnasium.wrappers import RecordEpisodeStatistics, ClipAction, NormalizeObservation, TransformObservation, NormalizeReward, TransformReward # pyright: ignore[reportPrivateImportUsage]
from pydantic import BaseModel, ConfigDict, Field, RootModel
import numpy as np # Imported so that it can be used by TransformObservation's `f` function

from big_rl.minigrid.envs import MinigridPreprocessing, MetaWrapper, ActionShuffle
from big_rl.mujoco.envs.wrappers import MujocoTaskRewardWrapper
from big_rl.utils.wrappers import PadObservation, PadAction, ToDictObservation, AddDummyObservation


WRAPPERS = [
    RecordEpisodeStatistics,
    ClipAction,
    NormalizeObservation,
    TransformObservation, # XXX: Can't do this with a YAML config. We need to be able to pass a lambda. We can interpret the 'f' kwarg using `eval`?
    NormalizeReward,
    TransformReward,
    # Atari
    AtariPreprocessing,
    FrameStack,
    # Minigrid
    MinigridPreprocessing,
    # Mujoco
    MujocoTaskRewardWrapper,
    # Misc
    MetaWrapper,
    ActionShuffle,
    TimeLimit,
    PadObservation,
    PadAction,
    ToDictObservation,
    AddDummyObservation,
]


WRAPPER_MAPPING = {
    wrapper.__name__: wrapper
    for wrapper in WRAPPERS
}


##################################################
# Config Model/Schema
##################################################


class EnvType(str, Enum):
    vector_env = 'VectorEnv'
    async_vector_env = 'AsyncVectorEnv'
    sync_vector_env = 'SyncVectorEnv'


class WrapperConfig(BaseModel):
    type: str
    kwargs: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra='forbid')


class SingleEnvConfig(BaseModel):
    kwargs: dict = {}
    repeat: int = Field(default=1, ge=1, description='Number of times to repeat the environment. Used for creating the EnvGroup containing this environment.')
    wrappers: list[WrapperConfig] = []
    name: str | None = Field(default=None, description='Human-readable name of the environment. Used for logging purposes. Multiple environments can share the same name if they are to be grouped together.')

    model_config = ConfigDict(extra='forbid')


class EnvGroupConfig(BaseModel):
    """ Defines how many environments to create and how to group them together. A group can be a single environment. """
    type: EnvType = EnvType.sync_vector_env
    envs: list[SingleEnvConfig] = []
    eval_only: bool = Field(default=False, description='If set to True, then the environment will not be used for training. Instead, we only run the model on it for evaluation purposes.')
    name: str | None = Field(default=None, description='Human-readable name of the environment or group of environments. Used for logging purposes, as well as for mapping submodels to groups of environments to train it on.')
    model_name: str | None = Field(default=None, description='Name of the submodel to use for this environment or group of environments. If not specified, then the parent model will be used.')

    model_config = ConfigDict(extra='forbid')


#EnvGroupGroupConfig = RootModel[list[EnvGroupConfig]]


##################################################
# Env Groups
##################################################


class EnvGroup(NamedTuple):
    env: gymnasium.vector.VectorEnv
    env_labels: list[str] # Names of the environments in `env`
    name: str | None # Name for this entire group of envs
    eval_only: bool
    model_name: str | None

    #def __getattribute__(self, __name: str) -> Any:
    #    if __name in ['step', 'reset', 'action_space', 'observation_space']:
    #        raise AttributeError(f'EnvGroup does not have attribute {__name}. It looks like you are trying to access an attribute of the underlying VectorEnv. You can access the underlying VectorEnv via the `env` attribute.')
    #    return super().__getattribute__(__name)


##################################################
# Env Creation
##################################################


def get_config_from_yaml(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    if isinstance(config, list):
        return [EnvGroupConfig.model_validate(c) for c in config]
    else:
        return [EnvGroupConfig.model_validate(config)]


def make_env_from_yaml(filename) -> list[EnvGroup]:
    config = get_config_from_yaml(filename)
    output = make_env(config)
    if isinstance(output, list):
        return output
    else:
        raise Exception()


def make_env(config : list[EnvGroupConfig] | list[dict]) -> list[EnvGroup]:
    if isinstance(config, list):
        return [make_env_group(c) for c in config]
    else:
        raise ValueError('Must pass in a list of config objects.')


def make_single_env(config : dict | SingleEnvConfig) -> gymnasium.Env:
    config = copy.deepcopy(config)
    if isinstance(config, dict):
        config = SingleEnvConfig.model_validate(config)
    if config.kwargs is None:
        raise ValueError('Must specify kwargs for `gymnasium.make()`.')
    kwargs = config.kwargs.copy()
    if 'id' not in kwargs:
        raise ValueError('Must specify the environment name (`id`) for `gymnasium.make()`.')
    env_id = kwargs.pop('id')
    env = gymnasium.make(env_id, **kwargs)
    wrapper_configs = config.wrappers
    for wrapper_config in wrapper_configs:
        wrapper_type = wrapper_config.type
        if wrapper_type is None:
            raise ValueError(f'Must specify wrapper type. Options are: {", ".join(WRAPPER_MAPPING.keys())}.')
        if wrapper_type not in WRAPPER_MAPPING:
            raise ValueError(f'Wrapper type {wrapper_type} not found. Options are: {", ".join(WRAPPER_MAPPING.keys())}.')
        # Handle special cases
        if wrapper_type == 'TransformObservation':
            wrapper_config.kwargs['f'] = eval(wrapper_config.kwargs['f'])
        # Default case
        env = WRAPPER_MAPPING[wrapper_type](env, **wrapper_config.kwargs)
    return env


def make_env_group(config: dict | EnvGroupConfig) -> EnvGroup:
    if isinstance(config, dict):
        config = EnvGroupConfig.model_validate(config)

    env_type = config.type
    match env_type:
        case None:
            raise ValueError('Must specify env type. Options are: Env, SyncVectorEnv, or AsyncVectorEnv.')
        case EnvType.vector_env:
            # For environments that are natively vector environments (i.e. not a bunch of regular environments wrapped in a [A]syncVectorEnv)
            raise NotImplementedError()
        case EnvType.sync_vector_env:
            vec_env_cls = SyncVectorEnv
        case EnvType.async_vector_env:
            vec_env_cls = AsyncVectorEnv
        case _:
            raise ValueError(f'Env type {env_type} not found. Options are: Env, SyncVectorEnv, or AsyncVectorEnv.')

    if config.envs is None or len(config.envs) == 0:
        raise ValueError('Must specify at least one environment.')

    env_configs = itertools.chain.from_iterable([
        [e]*e.repeat for e in config.envs
    ])

    return EnvGroup(
        env = vec_env_cls([lambda c=env_config: make_single_env(c) for env_config in env_configs]),
        eval_only = config.eval_only,
        name = config.name,
        model_name = config.model_name,
        env_labels = list(itertools.chain.from_iterable([[e.name or e.kwargs['id']] * e.repeat for e in config.envs]))
    )
