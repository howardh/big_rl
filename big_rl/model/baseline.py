from typing import List, Dict, Tuple, Any
from typing_extensions import Protocol # Needed for python<=3.7. Can import from typing in 3.8.

import torch
from torchtyping.tensor_type import TensorType

from big_rl.model.model import GreyscaleImageInput, ImageInput56, ScalarInput, DiscreteInput, LinearInput, MatrixInput, LinearOutput, StateIndependentOutput
from big_rl.model.recurrent_attention_16 import RecurrentAttention16


class BaselineModel(torch.nn.Module):
    def __init__(self, inputs, outputs, value_size, architecture = [32, 32]):
        super().__init__()
        self._value_size = value_size
        self._input_modules_config = inputs
        self._architecture = architecture

        input_size = sum(m.get('config',{}).get('value_size', value_size) for m in inputs.values()) # m['config']['value_size'] is the size of the outputs of each input module. Sum them up to get the total size of the concatenated input to the core ff module.
        architecture = [input_size, *architecture]

        self.input_modules = self._init_input_modules(inputs,
                key_size=1, value_size=value_size)
        self.output_modules = self._init_output_modules(outputs,
                key_size=architecture[-1], num_heads=1)

        self.ff_core = torch.nn.Sequential(
                *[torch.nn.Sequential(
                    torch.nn.Linear(a,b),
                    torch.nn.ReLU(),
                ) for a,b in zip(architecture[:-1], architecture[1:])],
        )

    def _init_input_modules(self, input_configs: Dict[str,Dict], key_size, value_size):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    GreyscaleImageInput,
                    ImageInput56,
                    ScalarInput,
                    DiscreteInput,
                    LinearInput,
                    MatrixInput,
                ]
        }
        input_modules: Dict[str,torch.nn.Module] = {}
        for k,v in input_configs.items():
            if v['type'] is None:
                input_modules[k] = v['module']
            else:
                if v['type'] not in valid_modules:
                    raise NotImplementedError(f'Unknown output module type: {v["type"]}')
                input_modules[k] = valid_modules[v['type']](**{
                        'key_size': key_size,
                        'value_size': value_size,
                        **v.get('config', {}),
                })
        return torch.nn.ModuleDict(input_modules)

    def _init_output_modules(self, output_configs: Dict[str,Dict], key_size, num_heads):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    LinearOutput,
                ]
        }
        output_modules: Dict[str,torch.nn.Module] = {}
        for k,v in output_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            if k == 'hidden':
                raise Exception('Cannot use "hidden" as an output module name')
            output_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = key_size,
                    num_heads = num_heads)
        return torch.nn.ModuleDict(output_modules)

    def forward(self,
            inputs: Dict[str,TensorType['batch_size','observation_shape']]):
        batch_size = next(iter(inputs.values())).shape[0]
        device = next(self.parameters()).device

        # Compute input to core module
        input_labels = []
        input_vals = []
        for k,module in sorted(self.input_modules.items()):
            module_config = self._input_modules_config[k]
            if 'inputs' in module_config:
                input_mapping = module_config['inputs']
                module_inputs = {}
                for dest_key, src_key in input_mapping.items():
                    if src_key not in inputs:
                        module_inputs = None
                        break
                    module_inputs[dest_key] = inputs[src_key]
                if module_inputs is not None:
                    y = module(**module_inputs)
                    input_labels.append(k)
                    input_vals.append(y['value'])
            else:
                if k not in inputs.keys():
                    # No data is provided, so fill with 0
                    y = torch.zeros([batch_size, self._input_modules_config[k].get('config',{}).get('value_size',self._value_size)], device=device) # XXX: not tested.
                else:
                    module_inputs = inputs[k]
                    y = module(module_inputs)['value']
                input_labels.append(k)
                input_vals.append(y)
        self.last_input_labels = input_labels

        values = torch.cat([
            *input_vals,
        ], dim=1)

        self.last_values = values

        # Core module computation
        ff_output = self.ff_core(values)

        # Compute output
        output = {}

        value = ff_output.view(1, batch_size, -1)
        key = torch.zeros_like(value, device=device)

        for k,v in self.output_modules.items():
            y = v(key, value)
            output[k] = y['output']

        return {
            **output,
            'misc': {
                #'core_output': layer_output,
                #'input_labels': input_labels,
            }
        }

    @property
    def has_attention(self):
        return False
