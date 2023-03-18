from typing import List, Dict, Tuple, Any
from typing_extensions import Protocol # Needed for python<=3.7. Can import from typing in 3.8.

import torch
from torchtyping.tensor_type import TensorType

from big_rl.model.model import GreyscaleImageInput, ImageInput56, ScalarInput, DiscreteInput, LinearInput, MatrixInput, LinearOutput, StateIndependentOutput
from big_rl.model.recurrent_attention_16 import RecurrentAttention16


class RecurrenceProtocol(Protocol):
    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        ...

    def forward(self, state: Tuple[torch.Tensor], key: torch.Tensor, value: torch.Tensor) -> Dict[str, Any]:
        ...

    def __call__(self, state: Tuple[torch.Tensor], key: torch.Tensor, value: torch.Tensor) -> Dict[str, torch.Tensor]:
        ...

    @property
    def num_outputs(self) -> int:
        """ Number of key-value pairs outputted by the model """
        ...

    @property
    def state_size(self) -> int:
        """ The number of elements in the state tuple. """
        ...


class ModularPolicy8(torch.nn.Module):
    """
    Same as ModularPolicy7 except the core modules are one module instead of being a list of modules.
    """
    def __init__(self, inputs, outputs, input_size, key_size, value_size, num_heads, recurrence_type='RecurrentAttention16', recurrence_kwargs={}):
        super().__init__()
        self._key_size = key_size
        self._input_size = input_size
        self._value_size = value_size
        self._input_modules_config = inputs

        self.input_modules = self._init_input_modules(inputs,
                key_size=key_size, value_size=value_size)
        self.output_modules = self._init_output_modules(outputs,
                key_size=key_size, num_heads=num_heads)

        self.attention = self._init_core_modules(
                recurrence_type = recurrence_type,
                input_size = input_size,
                key_size = key_size,
                value_size = value_size,
                num_heads = num_heads,
                recurrence_kwargs = recurrence_kwargs,
        )

        self.last_output = None

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
                input_modules[k] = valid_modules[v['type']](
                        **v.get('config', {}),
                        key_size = key_size,
                        value_size = value_size)
        return torch.nn.ModuleDict(input_modules)

    def _init_output_modules(self, output_configs: Dict[str,Dict], key_size, num_heads):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    LinearOutput,
                    StateIndependentOutput,
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

    def _init_core_modules(self, recurrence_type, input_size, key_size, value_size, num_heads, recurrence_kwargs) -> RecurrenceProtocol:
        recurrence_classes = {
                cls.__name__: cls
                for cls in [
                    RecurrentAttention16,
                ]
        }

        cls = None
        if recurrence_type in recurrence_classes:
            cls = recurrence_classes[recurrence_type]
        else:
            raise ValueError('Unknown recurrence type: {}'.format(recurrence_type))

        output = cls(input_size, key_size, value_size, num_heads, **recurrence_kwargs)
        return output

    def forward(self,
            inputs: Dict[str,TensorType['batch_size','observation_shape']],
            hidden: List[TensorType['num_blocks','batch_size','hidden_size']]):
        assert len(hidden) == 2+self.attention.state_size

        self.last_attention = []
        self.last_ff_gating = []
        self.last_output_attention = {}

        # Compute input to core module
        input_labels = []
        input_keys = []
        input_vals = []
        for k,module in self.input_modules.items():
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
                    input_keys.append(y['key'].unsqueeze(0))
                    input_vals.append(y['value'].unsqueeze(0))
            else:
                if k not in inputs:
                    continue # Skip this input module if no data is provided
                module_inputs = inputs[k]
                y = module(module_inputs)
                input_labels.append(k)
                input_keys.append(y['key'].unsqueeze(0))
                input_vals.append(y['value'].unsqueeze(0))
        self.last_input_labels = input_labels

        keys = torch.cat([
            *input_keys,
            hidden[0],
        ], dim=0)
        values = torch.cat([
            *input_vals,
            hidden[1],
        ], dim=0)

        self.last_keys = keys
        self.last_values = values

        core_state = tuple(hidden[2:])

        # Core module computation
        core_output = self.attention(core_state, keys, values)

        # Compute inputs to output modules
        new_keys = core_output['key']
        new_values = core_output['value']

        keys = torch.cat([
            *input_keys,
            new_keys,
        ], dim=0)
        values = torch.cat([
            *input_vals,
            new_values,
        ], dim=0)

        output = {}
        for k,v in self.output_modules.items():
            y = v(keys, values)
            output[k] = y['output']
            self.last_output_attention[k] = y['attn_output_weights'].cpu().detach().squeeze(1) # (batch_size, seq_len)

        self.last_hidden = (new_keys, new_values, *core_output['state'])

        #self.last_attention = core_output['misc']['attention'] # type: ignore
        #self.last_ff_gating = core_output['misc']['gates'] # type: ignore

        self.last_output = {
            **output,
            'hidden': self.last_hidden,
            'misc': {
                'core_output': core_output,
                'input_labels': input_labels,
                'output_attention': self.last_output_attention,
            }
        }

        return self.last_output

    def init_hidden(self, batch_size: int = 1):
        device = next(self.parameters()).device
        state = self.attention.init_state(batch_size)
        return (
                torch.zeros([self.attention.num_outputs, batch_size, self._key_size], device=device), # Key
                torch.zeros([self.attention.num_outputs, batch_size, self._key_size], device=device), # Value
                *state,
        )

    @property
    def has_attention(self):
        return False # TODO: Disabled drawing attention until it's implemented in `evalute_model.py`.
