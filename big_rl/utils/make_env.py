from enum import Enum
import itertools
import yaml

import gymnasium
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from gymnasium.wrappers import RecordEpisodeStatistics, ClipAction, TransformObservation, NormalizeReward, TransformReward # pyright: ignore[reportPrivateImportUsage]
from pydantic import BaseModel, ConfigDict

from big_rl.minigrid.envs import MinigridPreprocessing, MetaWrapper, ActionShuffle


WRAPPERS = [
    RecordEpisodeStatistics,
    ClipAction,
    TransformObservation, # XXX: Can't do this with a YAML config. We need to be able to pass a lambda. We can interpret the 'f' kwarg using `eval`?
    NormalizeReward,
    TransformReward,
    # Atari
    AtariPreprocessing,
    FrameStack,
    # Minigrid
    MinigridPreprocessing,
    # Misc
    MetaWrapper,
    ActionShuffle,
]


WRAPPER_MAPPING = {
    wrapper.__name__: wrapper
    for wrapper in WRAPPERS
}


##################################################
# Config Model/Schema
##################################################


class EnvType(str, Enum):
    Env = 'Env'
    vector_env = 'VectorEnv'
    async_vector_env = 'AsyncVectorEnv'
    sync_vector_env = 'SyncVectorEnv'


class WrapperConfig(BaseModel):
    type: str
    kwargs: dict

    model_config = ConfigDict(extra='forbid')


class SingleEnvConfig(BaseModel):
    kwargs: dict = {}
    repeat: int = 1
    wrappers: list[WrapperConfig] = []
    name: str | None = None

    model_config = ConfigDict(extra='forbid')


class EnvConfig(BaseModel):
    type: EnvType
    envs: list[SingleEnvConfig] = []

    model_config = ConfigDict(extra='forbid')


##################################################
# Env Creation
##################################################


def make_env_from_yaml(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    if isinstance(config, list):
        config = [EnvConfig.model_validate(c) for c in config]
    else:
        config  = EnvConfig.model_validate(config)
    return make_env(config)


def make_env(config : list | dict | EnvConfig):
    if isinstance(config, list):
        return [make_env(c) for c in config]
    elif isinstance(config, dict):
        config = EnvConfig.model_validate(config)
    assert isinstance(config, EnvConfig)

    env_type = config.type
    if env_type is None:
        raise ValueError('Must specify env type. Options are: Env, SyncVectorEnv, or AsyncVectorEnv.')
    if env_type == 'Env':
        return make_single_env(config.envs[0])
    if env_type == 'VectorEnv': # For environments that are natively vector environments (i.e. not a bunch of regular environments wrapped in a [A]syncVectorEnv)
        raise NotImplementedError()
    if env_type == 'SyncVectorEnv':
        vec_env_cls = SyncVectorEnv
    elif env_type == 'AsyncVectorEnv':
        vec_env_cls = AsyncVectorEnv
    else:
        raise ValueError(f'Env type {env_type} not found. Options are: Env, SyncVectorEnv, or AsyncVectorEnv.')

    if config.envs is None:
        raise ValueError()

    env_configs = itertools.chain.from_iterable([
        [e]*e.repeat for e in config.envs
    ])
    return vec_env_cls([lambda c=env_config: make_single_env(c) for env_config in env_configs])


def make_single_env(config : SingleEnvConfig):
    if config.kwargs is None:
        raise ValueError('Must specify kwargs for `gymnasium.make()`.')
    kwargs = config.kwargs.copy()
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


##################################################
# Env Labels
##################################################


def make_env_labels_from_yaml(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return make_env_labels(config)


def make_env_labels(config : list | dict | EnvConfig):
    if isinstance(config, list):
        return [make_env_labels(c) for c in config]
    elif isinstance(config, dict):
        config = EnvConfig.model_validate(config)
    assert isinstance(config, EnvConfig)

    env_type = config.type

    if env_type is None:
        raise ValueError('Must specify env type. Options are: Env, SyncVectorEnv, or AsyncVectorEnv.')

    if env_type == 'Env':
        return config.envs[0].kwargs['id']

    if env_type == 'VectorEnv': # For environments that are natively vector environments (i.e. not a bunch of regular environments wrapped in a [A]syncVectorEnv)
        raise NotImplementedError()

    if env_type in ['SyncVectorEnv', 'AsyncVectorEnv']:
        return list(itertools.chain.from_iterable([
            [e.name or e.kwargs['id']]*e.repeat for e in config.envs
        ]))

    raise ValueError(f'Env type {env_type} not found. Options are: Env, SyncVectorEnv, or AsyncVectorEnv.')
