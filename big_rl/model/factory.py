from collections import defaultdict
import copy
from typing import Dict
import inspect
import warnings
import logging

import torch
import pydantic
from pydantic import BaseModel, ConfigDict, FilePath, PositiveInt, Field

from big_rl.model.input_module.modules import IgnoredInput, GreyscaleImageInput, ImageInput56, ImageInput84, ScalarInput, UnaryScalarInput, DiscreteInput, LinearInput, MatrixInput
from big_rl.model.output_module.modules import LinearOutput, StateIndependentOutput
from big_rl.model.input_module.container import InputModuleContainer
from big_rl.model.core_module.container import CoreModule, CoreModuleContainer, CoreModuleParallel, CoreModuleSeries
from big_rl.model.output_module.container import OutputModuleContainer
from big_rl.model.modular_model_1 import ModularModel1
import big_rl.model.core_module
from big_rl.utils import merge_space
from big_rl.utils.make_env import EnvGroup


logger = logging.getLogger(__name__)


VALID_INPUT_MODULES = [
    IgnoredInput,
    GreyscaleImageInput,
    ImageInput56,
    ImageInput84,
    ScalarInput,
    UnaryScalarInput,
    DiscreteInput,
    LinearInput,
    MatrixInput,
]


VALID_OUTPUT_MODULES = [
    LinearOutput,
    StateIndependentOutput,
]


##################################################
# Config Model/Schema
##################################################


class WeightConfig(BaseModel):
    filename: FilePath | None = Field(
            default=None,
            description='Path to a file containing the weights for this module.'
    )
    key_prefix: str = Field(
            default='',
            description='Prefix to add to the keys in the state dict. Useful for loading weights from a model with a different architecture.'
    )
    freeze: bool = Field(
            default=False,
            description='Freeze the weights of this module. This is useful for transfer learning.'
    )

    model_config = ConfigDict(extra='forbid')


class PeripheralModuleConfig(BaseModel):
    type: str
    kwargs: dict = {}
    key_size: PositiveInt | None = None
    value_size: PositiveInt | None = None
    num_heads: PositiveInt | None = None
    weight_config: WeightConfig | None = None

    model_config = ConfigDict(extra='forbid')


class CoreModuleConfig(BaseModel):
    type: str | None = None
    kwargs: dict = {}
    key_size: PositiveInt | None = None
    value_size: PositiveInt | None = None
    num_heads: PositiveInt | None = None
    weight_config: WeightConfig | None = None

    container: str | None = None
    modules: list['CoreModuleConfig'] = []

    model_config = ConfigDict(extra='forbid')

    @pydantic.model_validator(mode='after')
    def check_module_or_container(self):
        if self.type is None and self.container is None:
            raise ValueError('Must specify either a module type ("type") or container type ("container") for a core module.')
        elif self.type is not None and self.container is not None:
            raise ValueError('Cannot specify both a module type ("type") and container type ("container") for a core module.')
        
        if self.container is None and len(self.modules) > 0:
            raise ValueError('Cannot specify children modules ("modules") if container type ("container") is not specified for a core module.')

        return self


class SubModelConfig(BaseModel):
    input_modules: dict[str, str] = Field(
            default={},
            description='Mapping from input module key/name to input name',
    )
    output_modules: dict[str, str] = Field(
            default={},
            description='Mapping from output module key/name to output name',
    )

    model_config = ConfigDict(extra='forbid')


class ModelConfig(BaseModel):
    type: str
    key_size: PositiveInt = 512
    value_size: PositiveInt = 512
    num_heads: PositiveInt = 8
    input_modules: dict[str, PeripheralModuleConfig]
    output_modules: dict[str, PeripheralModuleConfig]
    core_modules: CoreModuleConfig
    weight_config: WeightConfig | None = None

    submodel_configs: dict[str, SubModelConfig] | None = Field(
        default=None,
        description='Used to define multiple submodels that share the same core modules with each other and the parent model. Each submodel can have its own input and output modules, but they must share the same core modules.',
    )

    model_config = ConfigDict(extra='forbid')


##################################################
# Model Creation
##################################################


def init_weights(module, weight_config: WeightConfig | None, strict=True):
    if weight_config is None:
        return
    if weight_config.filename is not None:
        key_prefix = weight_config.key_prefix
        if key_prefix is None:
            raise ValueError('Must specify a key prefix if you want to load weights from a file')
        state_dict = torch.load(weight_config.filename,
                                map_location=torch.device('cpu'))
        state_dict = state_dict['model']

        # Filter keys
        if key_prefix == '':
            filtered_state_dict = state_dict
        else:
            filtered_keys = [k for k in state_dict.keys() if k.startswith(key_prefix)]
            if len(filtered_keys) == 0:
                raise ValueError(f'Key prefix "{key_prefix}" not found in state dict')
            filtered_state_dict = {
                    k[len(key_prefix)+1:]:v
                    for k,v in state_dict.items() if k in filtered_keys
            }
        # Load weights
        module.load_state_dict(filtered_state_dict, strict=strict)


    # Freeze weights if requested
    if weight_config.freeze:
        for param in module.parameters():
            param.requires_grad = False
    else:
        for param in module.parameters():
            param.requires_grad = True


def create_model(config: dict | ModelConfig, observation_space=None, action_space=None, envs: list[EnvGroup] | None = None):
    if isinstance(config, dict):
        config = ModelConfig.model_validate(config)
    if envs is not None:
        spaces_by_module = _get_space_by_module(envs, config)
    else:
        spaces_by_module = {
            'input': {'observation_space': observation_space, 'action_space': action_space},
            'output': {'observation_space': observation_space, 'action_space': action_space},
        }

    #model_type = config.get('type')
    #key_size = config.get('key_size', 512)
    #value_size = config.get('value_size', 512)
    #num_heads = config.get('num_heads', 8)

    if config.type is None:
        raise ValueError('Model type must be specified in config')

    if config.type == 'ModularModel1':
        # Initialize model architecture
        input_modules = create_input_modules(
                config=config.input_modules,
                key_size=config.key_size,
                value_size=config.value_size,
                observation_space=spaces_by_module['input']['observation_space'],
                action_space=spaces_by_module['input']['action_space'],
        )
        core_modules = create_core_modules(
                config=config.core_modules,
                key_size=config.key_size,
                value_size=config.value_size,
                num_heads=config.num_heads,
        )
        output_modules = create_output_modules(
                config=config.output_modules,
                key_size=config.key_size,
                value_size=config.value_size,
                num_heads=config.num_heads,
                observation_space=spaces_by_module['output']['observation_space'],
                action_space=spaces_by_module['output']['action_space'],
        )

        model = ModularModel1(
                input_modules=InputModuleContainer(input_modules),
                core_modules=core_modules,
                output_modules=OutputModuleContainer(output_modules),
                key_size=config.key_size,
                value_size=config.value_size,
                num_heads=config.num_heads,
                submodel_configs=config.submodel_configs,
        )

        # Intialize weights
        # This needs to be done separately in case we want to load weights for the entire model and then overwrite the weights of some constituent modules
        init_weights(model, config.weight_config, strict=False)
        for k,m in model.input_modules.input_modules.items():
            init_weights(m, config.input_modules[k].weight_config)
        for k,m in model.output_modules.output_modules.items():
            init_weights(m, config.output_modules[k].weight_config)
        def init_core_weights(module, config):
            if config.weight_config is not None:
                init_weights(module, config.weight_config)
            if isinstance(module, CoreModuleContainer):
                for m,c in zip(module.core_modules, config.modules):
                    init_core_weights(m, c)
        init_core_weights(model.core_modules, config.core_modules)

        return model
    else:
        raise NotImplementedError()


def create_input_modules(config: dict[str,dict] | dict[str,PeripheralModuleConfig], key_size: int | None = None, value_size: int | None = None, observation_space=None, action_space=None) -> torch.nn.ModuleDict:
    if len(config) == 0:
        return torch.nn.ModuleDict()
    if isinstance(next(iter(config.values())), dict):
        config = {
            k:PeripheralModuleConfig.model_validate(v)
            for k,v in config.items()
        }
    config = copy.deepcopy(config)

    valid_modules = {
            cls.__name__: cls
            for cls in VALID_INPUT_MODULES
    }
    input_modules: Dict[str,torch.nn.Module] = {}
    for module_name,module_config in config.items():
        assert isinstance(module_config, PeripheralModuleConfig)

        class_name = module_config.type
        key_size = module_config.key_size or key_size
        value_size = module_config.value_size or value_size

        # Check if we requested a valid module
        if class_name not in valid_modules:
            raise NotImplementedError(f'Unknown output module type: {module_config.type}')

        # Initialize model
        module_config.kwargs = _preprocess_kwargs(
                kwargs = module_config.kwargs,
                cls = valid_modules[module_config.type],
                observation_space = observation_space[module_name] if isinstance(observation_space, dict) else observation_space,
                action_space = action_space[module_name] if isinstance(action_space, dict) else action_space,
        )

        logger.debug(f'Creating input module {module_name} of type {class_name} with kwargs {module_config.kwargs}')

        input_modules[module_name] = valid_modules[class_name](
                **module_config.kwargs,
                key_size = key_size,
                value_size = value_size,
        )
        
        # Weight config
        #init_weights()
        #weight_config = module_config.get('weight_config', {})

        ## Load weights if requested
        #if 'filename' in weight_config:
        #    key_prefix = weight_config.get('key_prefix')
        #    if key_prefix is None:
        #        raise ValueError('Must specify a key prefix if you want to load weights from a file')
        #    state_dict = torch.load(weight_config['filename'])
        #    # Filter keys
        #    filtered_keys = [k for k in state_dict.keys() if k.startswith(key_prefix)]
        #    if len(filtered_keys) == 0:
        #        raise ValueError(f'Key prefix "{key_prefix}" not found in state dict')
        #    filtered_state_dict = {
        #            k[len(key_prefix)+1:]:v
        #            for k,v in state_dict.items() if k in filtered_keys
        #    }
        #    # Load weights
        #    input_modules[module_name].load_state_dict(filtered_state_dict)

        ## Freeze weights if requested
        #if weight_config.get('freeze', False):
        #    for param in input_modules[module_name].parameters():
        #        param.requires_grad = False

    return torch.nn.ModuleDict(input_modules)


def create_output_modules(config: dict, key_size: int | None = None, value_size: int | None = None, num_heads: int | None = None, observation_space=None, action_space=None):
    if len(config) == 0:
        return torch.nn.ModuleDict()
    if isinstance(next(iter(config.values())), dict):
        config = {
            k:PeripheralModuleConfig.model_validate(v)
            for k,v in config.items()
        }
    config = copy.deepcopy(config)
    valid_modules = {
            cls.__name__: cls
            for cls in VALID_OUTPUT_MODULES
    }

    output_modules: Dict[str,torch.nn.Module] = {}
    for module_name,module_config in config.items():
        assert isinstance(module_config, PeripheralModuleConfig)

        class_name = module_config.type
        #key_size = module_config.get('key_size', key_size)
        #value_size = module_config.get('value_size', value_size)
        #num_heads = module_config.get('num_heads', num_heads)

        if class_name not in valid_modules:
            raise NotImplementedError(f'Unknown output module type: {module_config.type}')
        if module_name == 'hidden':
            raise Exception(f'Cannot use "{module_name}" as an output module name')

        module_config.kwargs = _preprocess_kwargs(
                kwargs = module_config.kwargs,
                cls = valid_modules[class_name],
                observation_space = observation_space[module_name] if isinstance(observation_space, dict) else observation_space,
                action_space = action_space[module_name] if isinstance(action_space, dict) else action_space,
        )

        output_modules[module_name] = valid_modules[class_name](
                **module_config.kwargs,
                key_size = key_size,
                value_size = value_size,
                num_heads = num_heads
        )
    return torch.nn.ModuleDict(output_modules)


def create_core_modules(config: dict | CoreModuleConfig, key_size=None, value_size=None, num_heads=None) -> CoreModule:
    if isinstance(config, dict):
        config = CoreModuleConfig.model_validate(config)

    container_type = config.container
    class_name = config.type
    key_size = config.key_size or key_size
    value_size = config.value_size or value_size
    num_heads = config.num_heads or num_heads

    if container_type is not None:
        if container_type == 'parallel':
            submodule_configs = config.modules
            return CoreModuleParallel([
                create_core_modules(
                    submodule_config,
                    key_size = key_size,
                    value_size = value_size,
                    num_heads = num_heads,
                )
                for submodule_config in submodule_configs
            ])
        elif container_type == 'series':
            submodule_configs = config.modules
            return CoreModuleSeries([
                create_core_modules(
                    submodule_config,
                    key_size = key_size,
                    value_size = value_size,
                    num_heads = num_heads,
                )
                for submodule_config in submodule_configs
            ])
        else:
            raise ValueError(f'Unknown container type: {container_type}')

    if class_name is None and container_type is None:
        raise ValueError('Container ("parallel" or "series") or module type must be specified in config')
    if key_size is None:
        raise ValueError('Key size must be specified in config')
    if value_size is None:
        raise ValueError('Value size must be specified in config')
    if num_heads is None:
        raise ValueError('Number of heads must be specified in config')

    valid_modules = {
            cls.__name__: cls
            for cls in CoreModule.subclasses
    }

    cls = None
    if class_name in valid_modules:
        cls = valid_modules[class_name]
    else:
        raise ValueError('Unknown recurrence type: {}'.format(class_name))

    output = cls(
            key_size=key_size,
            value_size=value_size,
            num_heads=num_heads,
            **config.kwargs
    )
    return output


def _preprocess_kwargs(kwargs, cls, observation_space=None, action_space=None):
    # Check that the types match with the function signature type annotations
    # Perform pre-processing on the configs if necessary (e.g. extracting values from the observation/action space)
    config = copy.deepcopy(kwargs)

    params = inspect.signature(cls).parameters
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    for param_name, param_value in config.items():
        if param_name not in params and not has_kwargs:
            raise ValueError(f'Invalid parameter for module {cls.__name__}: {param_name}')
        expected_type = params[param_name].annotation
        if expected_type is inspect.Parameter.empty:
            continue
        if not isinstance(param_value, expected_type):
            # If there's a type mismatch, check if we can convert the value
            # A dictionary of the format `{'source': 'action/observation_space', 'accessor': 'shape[0]'}` means that the value is to be extracted from the action or observation space.
            if isinstance(param_value, dict) and 'source' in param_value and 'accessor' in param_value:
                if param_value['source'] in ['action_space', 'observation_space']:
                    # Make sure the source is not None
                    if param_value['source'] == 'action_space' and action_space is None:
                        raise ValueError(f'Value requested from action space but no action space was provided')
                    if param_value['source'] == 'observation_space' and observation_space is None:
                        raise ValueError(f'Value requested from observation space but no observation space was provided')
                    # XXX: Dangerous. Warn user in case someone other than me tries to run this code and they're using configurations from an untrusted source. Could be fixed by parsing out the accessor and using `getattr()` and `__getitem__()`, but not worth the effort when I'm the only person using this.
                    warnings.warn('This piece of code is potentially dangerous. Do not run if the model configuration does not come from a trusted source.')
                    config[param_name] = eval(f'{param_value["source"]}{param_value["accessor"]}')
                else:
                    raise ValueError(f'Unknown source: {param_value["source"]}. Valid sources are "action_space" and "observation_space".')
            else:
                raise TypeError(f'Invalid type for parameter {param_name} of module {cls.__name__}. Expected {expected_type}. Received type {type(param_value)} ({param_value}).')

    return config


def _get_space_by_module(envs: list[EnvGroup], model_config: ModelConfig):
    """ Return mappings from peripheral module name to the corresponding observation/action space. """
    # For each module, find the submodels it belongs to
    # For each submodel, find the environments associated with it and their corresponding observation/action spaces
    if model_config.submodel_configs is None:
        # TODO: Merge all observation/action spaces together and map all modules to the same spaces
        observation_spaces = merge_space(*[env.env.single_observation_space for env in envs])
        action_spaces = merge_space(*[env.env.single_action_space for env in envs])
        return {
            'input': {
                'observation_space': observation_spaces,
                'action_space': action_spaces,
            },
            'output': {
                'observation_space': observation_spaces,
                'action_space': action_spaces,
            },
        }
    else:
        input_modules_to_submodels = defaultdict(lambda: [])
        output_modules_to_submodels = defaultdict(lambda: [])
        for submodel_name, submodel_config in model_config.submodel_configs.items():
            for input_module_name in submodel_config.input_modules:
                input_modules_to_submodels[input_module_name].append(submodel_name)
            for output_module_name in submodel_config.output_modules:
                output_modules_to_submodels[output_module_name].append(submodel_name)

        submodel_to_observation_space = {}
        submodel_to_action_space = {}
        for env in envs:
            submodel_to_observation_space[env.model_name] = env.env.single_observation_space
            submodel_to_action_space[env.model_name] = env.env.single_action_space

        input_module_to_observation_space = defaultdict(lambda: [])
        input_module_to_action_space = defaultdict(lambda: [])
        output_module_to_observation_space = defaultdict(lambda: [])
        output_module_to_action_space = defaultdict(lambda: [])
        for module_name, submodel_names in input_modules_to_submodels.items():
            for submodel_name in submodel_names:
                input_module_to_observation_space[module_name].append(submodel_to_observation_space[submodel_name])
                input_module_to_action_space[module_name].append(submodel_to_action_space[submodel_name])
        for module_name, submodel_names in output_modules_to_submodels.items():
            for submodel_name in submodel_names:
                output_module_to_observation_space[module_name].append(submodel_to_observation_space[submodel_name])
                output_module_to_action_space[module_name].append(submodel_to_action_space[submodel_name])

        input_module_to_observation_space = {k: merge_space(*v) for k,v in input_module_to_observation_space.items()}
        input_module_to_action_space = {k: merge_space(*v) for k,v in input_module_to_action_space.items()}
        output_module_to_observation_space = {k: merge_space(*v) for k,v in output_module_to_observation_space.items()}
        output_module_to_action_space = {k: merge_space(*v) for k,v in output_module_to_action_space.items()}

        return {
            'input': {
                'observation_space': input_module_to_observation_space,
                'action_space': input_module_to_action_space
            },
            'output': {
                'observation_space': output_module_to_observation_space,
                'action_space': output_module_to_action_space
            },
        }
