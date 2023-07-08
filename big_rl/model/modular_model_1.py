import torch

from big_rl.model.core_module.container import CoreModule
from big_rl.model.input_module.container import InputModuleContainer
from big_rl.model.output_module.container import OutputModuleContainer

class ModularModel1(torch.nn.Module):
    def __init__(self, input_modules: InputModuleContainer, core_modules: CoreModule, output_modules: OutputModuleContainer, key_size: int, value_size: int, num_heads: int):
        super().__init__()
        self.input_modules = input_modules
        self.core_modules = core_modules
        self.output_modules = output_modules

        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads

        self._initial_state = torch.nn.ParameterDict({
                'key': torch.nn.Parameter(torch.zeros(1, self._key_size)),
                'value': torch.nn.Parameter(torch.zeros(1, self._value_size)),
        })

    def forward(self, x, hidden):
        prev_key = hidden[0]
        prev_value = hidden[1]

        x1 = self.input_modules(x)

        x2 = self.core_modules(
                key = torch.cat([*x1['key'], prev_key], dim=0),
                value = torch.cat([*x1['value'], prev_value], dim=0),
                hidden = hidden[2:]
        )

        # Join key and values from x1 and x2
        key = torch.cat([*x1['key'], x2['key']], dim=0)
        value = torch.cat([*x1['value'], x2['value']], dim=0)

        x3 = self.output_modules(key, value)

        return {
            'hidden': tuple([x2['key'], x2['value'], *x2['hidden']]),
            **x3,
        }

    def init_hidden(self, batch_size):
        key = self._initial_state['key'].expand(batch_size, -1).view(1, batch_size, -1)
        value = self._initial_state['value'].expand(batch_size, -1).view(1, batch_size, -1)
        initial_hidden = self.core_modules.init_hidden(batch_size)
        initial_output = self.core_modules(key, value, initial_hidden)
        return tuple([
            initial_output['key'],
            initial_output['value'],
            *initial_output['hidden'],
        ])

    @property
    def n_hidden(self):
        return self.core_modules.n_hidden() + 2
