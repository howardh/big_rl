import torch

from big_rl.model.core_module.container import CoreModule
from big_rl.model.input_module.container import InputModuleContainer
from big_rl.model.output_module.container import OutputModuleContainer


class ModularModel1(torch.nn.Module):
    def __init__(self, input_modules: InputModuleContainer, core_modules: CoreModule, output_modules: OutputModuleContainer, key_size: int, value_size: int, num_heads: int, submodel_configs: dict | None = None):
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

        if submodel_configs is not None:
            self.submodels = self.init_submodels(submodel_configs)

    def forward(self, x, hidden):
        prev_key = hidden[0]
        prev_value = hidden[1]

        input_hidden, core_hidden, output_hidden = self._split_hidden(hidden[2:])

        x1 = self.input_modules(x, input_hidden)

        x2 = self.core_modules(
                key = torch.cat([*x1['key'], prev_key], dim=0),
                value = torch.cat([*x1['value'], prev_value], dim=0),
                hidden = core_hidden,
        )

        # Join key and values from x1 and x2
        key = torch.cat([*x1['key'], x2['key']], dim=0)
        value = torch.cat([*x1['value'], x2['value']], dim=0)

        x3 = self.output_modules(key, value, output_hidden)

        return {
            'hidden': tuple([
                x2['key'],
                x2['value'],
                *x1['hidden'],
                *x2['hidden'],
                *x3.pop('hidden', []),
            ]),
            **x3,
            'misc': {
                'input': x1.get('misc', {}),
                'core': x2.get('misc', {}),
                'output': x3.get('misc', {}),
            },
        }

    def init_hidden(self, batch_size):
        key = self._initial_state['key'].expand(batch_size, -1).view(1, batch_size, -1)
        value = self._initial_state['value'].expand(batch_size, -1).view(1, batch_size, -1)
        initial_hidden_input = self.input_modules.init_hidden(batch_size)
        initial_hidden_core = self.core_modules.init_hidden(batch_size)
        initial_hidden_output = self.output_modules.init_hidden(batch_size)
        initial_output = self.core_modules(key, value, initial_hidden_core)
        output = tuple([
            initial_output['key'],
            initial_output['value'],
            *initial_hidden_input,
            *initial_output['hidden'],
            *initial_hidden_output,
        ])
        assert len(output) == self.n_hidden
        return output

    @property
    def n_hidden(self):
        return 2 + self.input_modules.n_hidden + self.core_modules.n_hidden + self.output_modules.n_hidden

    @property
    def hidden_batch_dims(self):
        return [1, 1] + self.input_modules.hidden_batch_dims + self.core_modules.hidden_batch_dims + self.output_modules.hidden_batch_dims

    def init_submodels(self, submodel_configs):
        submodels = {}
        for submodel_name, submodel_config in submodel_configs.items():
            submodels[submodel_name] = ModularModel1(
                    input_modules=InputModuleContainer(torch.nn.ModuleDict({
                        input_name: self.input_modules.input_modules[module_name]
                        for module_name, input_name in submodel_config.input_modules.items()
                    })),
                    core_modules=self.core_modules.submodel(submodel_name),
                    output_modules=OutputModuleContainer(torch.nn.ModuleDict({
                        output_name: self.output_modules.output_modules[module_name]
                        for module_name, output_name in submodel_config.output_modules.items()
                    })),
                    key_size=self._key_size,
                    value_size=self._value_size,
                    num_heads=self._num_heads,
            )
            submodels[submodel_name]._initial_state = self._initial_state
        return submodels

    def __getitem__(self, key):
        if not hasattr(self, 'submodels'):
            raise Exception('No submodels defined.')
        return self.submodels[key]

    def _split_hidden(self, hidden):
        input_hidden_size = self.input_modules.n_hidden
        core_hidden_size = self.core_modules.n_hidden
        output_hidden_size = self.output_modules.n_hidden

        input_hidden = hidden[:input_hidden_size]
        core_hidden = hidden[input_hidden_size:input_hidden_size + core_hidden_size]
        output_hidden = hidden[input_hidden_size + core_hidden_size:]

        assert len(input_hidden) == input_hidden_size
        assert len(core_hidden) == core_hidden_size
        assert len(output_hidden) == output_hidden_size

        return input_hidden, core_hidden, output_hidden
