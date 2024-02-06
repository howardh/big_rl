import itertools
from typing import Any

import torch


class OutputModuleContainer(torch.nn.Module):
    def __init__(self, output_modules: torch.nn.ModuleDict):
        super().__init__()

        self.output_modules = output_modules
        self._module_order = sorted(list(output_modules.keys()))

        if 'misc' in self.output_modules:
            raise ValueError('Output module name "misc" is reserved')

    def forward(self, key, value, hidden=tuple()):
        hidden_by_module = self._split_hidden(hidden)
        new_hidden_by_module = {}
        output: dict[str,Any] = {'misc': {}}
        for k,module in self.output_modules.items():
            if hasattr(module, 'n_hidden'):
                o = module(key, value, hidden_by_module[k])
            else:
                o = module(key, value)
            output[k] = o['output']
            output['misc'][k] = o.get('misc', None)
            if 'hidden' in o:
                new_hidden_by_module[k] = o['hidden']

        # If a module was not called, then maintain its hidden state unchanged
        for k in self._module_order:
            if k not in new_hidden_by_module:
                new_hidden_by_module[k] = hidden_by_module[k]
        if len(new_hidden_by_module) > 0:
            output['hidden'] = tuple(itertools.chain(*[new_hidden_by_module[k] for k in self._module_order]))
        else:
            output['hidden'] = tuple()

        return output

    @property
    def n_hidden(self) -> int:
        return sum(
            m.n_hidden if hasattr(m, 'n_hidden') else 0
            for m in self.output_modules.values()
        )

    @property
    def hidden_batch_dims(self) -> list[int]:
        return list(itertools.chain.from_iterable([
            self.output_modules[k].hidden_batch_dims
            for k in self._module_order
            if hasattr(self.output_modules[k], 'hidden_batch_dims')
        ]))

    def init_hidden(self, batch_size: int) -> tuple:
        return tuple(itertools.chain.from_iterable([
            self.output_modules[k].init_hidden(batch_size)
            for k in self._module_order
            if hasattr(self.output_modules[k], 'init_hidden')
        ]))

    def _split_hidden(self, hidden):
        i = 0
        split_hidden = {}
        for k in self._module_order:
            if hasattr(self.output_modules[k], 'n_hidden'):
                n = self.output_modules[k].n_hidden
                split_hidden[k] = hidden[i:i+n]
                i += n
            else:
                split_hidden[k] = tuple()
        return split_hidden
