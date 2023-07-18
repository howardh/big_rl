import itertools
import copy
from typing import Dict
import inspect
import warnings
import logging

import torch

from big_rl.model.input_module.modules import IgnoredInput, GreyscaleImageInput, ImageInput56, ImageInput84, ScalarInput, DiscreteInput, LinearInput, MatrixInput
from big_rl.model.output_module.modules import LinearOutput, StateIndependentOutput
from big_rl.model.input_module.container import InputModuleContainer
from big_rl.model.core_module.container import CoreModule, CoreModuleContainer, CoreModuleParallel, CoreModuleSeries
from big_rl.model.output_module.container import OutputModuleContainer
from big_rl.model.modular_model_1 import ModularModel1
from big_rl.model.core_module.recurrent_attention_17 import RecurrentAttention17


logger = logging.getLogger(__name__)


def init_weights(module, weight_config, strict=True):
    if weight_config is None:
        return
    if 'filename' in weight_config:
        key_prefix = weight_config.get('key_prefix')
        if key_prefix is None:
            raise ValueError('Must specify a key prefix if you want to load weights from a file')
        state_dict = torch.load(weight_config['filename'],
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
    freeze = weight_config.get('freeze')
    if freeze is not None:
        if freeze:
            for param in module.parameters():
                param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = True


def create_model(config, observation_space=None, action_space=None):
    model_type = config.get('type')
    key_size = config.get('key_size', 512)
    value_size = config.get('value_size', 512)
    num_heads = config.get('num_heads', 8)

    if model_type is None:
        raise ValueError('Model type must be specified in config')

    if model_type == 'ModularModel1':
        # Initialize model architecture
        input_modules = create_input_modules(
                config=config['input_modules'],
                key_size=key_size,
                value_size=value_size,
                observation_space=observation_space,
                action_space=action_space,
        )
        core_modules = create_core_modules(
                config=config['core_modules'],
                key_size=key_size,
                value_size=value_size,
                num_heads=num_heads,
        )
        output_modules = create_output_modules(
                config=config['output_modules'],
                key_size=key_size,
                value_size=value_size,
                num_heads=num_heads,
                observation_space=observation_space,
                action_space=action_space,
        )

        model = ModularModel1(
                input_modules=InputModuleContainer(input_modules),
                core_modules=core_modules,
                output_modules=OutputModuleContainer(output_modules),
                key_size=key_size,
                value_size=value_size,
                num_heads=num_heads,
        )

        # Intialize weights
        # This needs to be done separately in case we want to load weights for the entire model and then overwrite the weights of some constituent modules
        init_weights(model, config.get('weight_config'), strict=False)
        for k,m in model.input_modules.input_modules.items():
            init_weights(m, config['input_modules'][k].get('weight_config'))
        for k,m in model.output_modules.output_modules.items():
            init_weights(m, config['output_modules'][k].get('weight_config'))
        def init_core_weights(module, config):
            if 'weight_config' in config:
                init_weights(module, config['weight_config'])
            if isinstance(module, CoreModuleContainer):
                for m,c in zip(module.core_modules, config['modules']):
                    init_core_weights(m, c)
        init_core_weights(model.core_modules, config['core_modules'])

        return model
    else:
        raise NotImplementedError()


def create_input_modules(config: Dict[str,Dict], key_size: int | None = None, value_size: int | None = None, observation_space=None, action_space=None) -> torch.nn.ModuleDict:
    config = copy.deepcopy(config)
    valid_modules = {
            cls.__name__: cls
            for cls in [
                IgnoredInput,
                GreyscaleImageInput,
                ImageInput56,
                ImageInput84,
                ScalarInput,
                DiscreteInput,
                LinearInput,
                MatrixInput,
            ]
    }
    input_modules: Dict[str,torch.nn.Module] = {}
    for module_name,module_config in config.items():
        class_name = module_config.get('type')
        key_size = module_config.get('key_size', key_size)
        value_size = module_config.get('value_size', value_size)

        # Check if we requested a valid module
        if class_name not in valid_modules:
            raise NotImplementedError(f'Unknown output module type: {module_config["type"]}')

        # Initialize model
        module_config['kwargs'] = _preprocess_kwargs(
                kwargs = module_config.get('kwargs',{}),
                cls = valid_modules[module_config['type']],
                observation_space = observation_space,
                action_space = action_space
        )

        logger.debug(f'Creating input module {module_name} of type {class_name} with kwargs {module_config["kwargs"]}')

        input_modules[module_name] = valid_modules[class_name](
                **module_config['kwargs'],
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


def create_output_modules(config, key_size: int | None = None, value_size: int | None = None, num_heads: int | None = None, observation_space=None, action_space=None):
    config = copy.deepcopy(config)
    valid_modules = {
            cls.__name__: cls
            for cls in [
                LinearOutput,
                StateIndependentOutput,
            ]
    }

    output_modules: Dict[str,torch.nn.Module] = {}
    for module_name,module_config in config.items():
        class_name = module_config.get('type')
        key_size = module_config.get('key_size', key_size)
        value_size = module_config.get('value_size', value_size)
        num_heads = module_config.get('num_heads', num_heads)

        if class_name not in valid_modules:
            raise NotImplementedError(f'Unknown output module type: {module_config["type"]}')
        if module_name == 'hidden':
            raise Exception(f'Cannot use "{module_name}" as an output module name')

        module_config['kwargs'] = _preprocess_kwargs(
                kwargs = module_config.get('kwargs',{}),
                cls = valid_modules[class_name],
                observation_space = observation_space,
                action_space = action_space
        )

        output_modules[module_name] = valid_modules[class_name](
                **module_config.get('kwargs', {}),
                key_size = key_size,
                value_size = value_size,
                num_heads = num_heads
        )
    return torch.nn.ModuleDict(output_modules)


def create_core_modules(config, key_size=None, value_size=None, num_heads=None) -> CoreModule:
    container_type = config.get('container')
    class_name = config.get('type')
    key_size = config.get('key_size', key_size)
    value_size = config.get('value_size', value_size)
    num_heads = config.get('num_heads', num_heads)
    kwargs = config.get('kwargs', {})

    if container_type is not None:
        if container_type == 'parallel':
            submodule_configs = config.get('modules', [])
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
            submodule_configs = config.get('modules', [])
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
            **kwargs
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
