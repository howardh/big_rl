import itertools

import torch

from .container import CoreModule
from .base_attention import BaseAttentionCoreModule


class HeadlessContainer(BaseAttentionCoreModule):
    def __init__(self, key_size, value_size, num_heads, models: list[torch.nn.Module] = [], decapitate_kwargs: dict = {}):
        super().__init__(key_size, value_size, num_heads)

        self.submodels = torch.nn.ModuleList()
        self.ff_in = torch.nn.ModuleList()
        self.ff_out = torch.nn.ModuleList()

        for model in models:
            if not hasattr(model, 'decapitate'):
                raise ValueError(f'Model of type {type(model)} does not support decapitation.')
            model = model.decapitate(**decapitate_kwargs)
            if not hasattr(model, 'input_size'):
                raise ValueError(f'Decapitated model of type {type(model)} does not have an input_size attribute.')
            if not hasattr(model, 'output_size'):
                raise ValueError(f'Decapitated model of type {type(model)} does not have an output_size attribute.')
            
            self.submodels.append(model)

            # Stuff around the headless model
            self.ff_in.append(torch.nn.Linear(value_size, model.input_size))
            self.ff_out.append(torch.nn.Linear(model.output_size, key_size + value_size))

    def compute_output(self, attn_output, hidden):
        if len(self.submodels) == 0:
            batch_size = attn_output.shape[0]
            return {
                'key': torch.empty(0, batch_size, self._key_size),
                'value': torch.empty(0, batch_size, self._value_size),
                'hidden': tuple(),
                'misc': {},
            }

        hidden_iter = iter(hidden)
        split_hidden = [
                list(itertools.islice(hidden_iter, m.n_hidden))
                for m in self.submodels
        ]

        new_key = []
        new_value = []
        new_hidden = []

        for m,ff_in,ff_out,h in zip(self.submodels, self.ff_in, self.ff_out, split_hidden):
            x = attn_output
            x = ff_in(x)

            model_output = m(x, h)
            x = model_output['output']

            x = ff_out(x)
            key, value = torch.split(x, [self._key_size, self._value_size], dim=-1)

            new_key.append(key)
            new_value.append(value)
            new_hidden.append(model_output['hidden'])

        return {
            'key': torch.stack(new_key, dim=0),
            'value': torch.stack(new_value, dim=0),
            'hidden': tuple(itertools.chain.from_iterable(new_hidden)),
            'misc': {}, # TODO: ???
        }

    def init_hidden(self, batch_size) -> tuple[torch.Tensor, ...]:
        return tuple(itertools.chain.from_iterable([m.init_hidden(batch_size) for m in self.submodels]))

    @property
    def n_hidden(self):
        return sum(m.n_hidden for m in self.submodels)

    @property
    def hidden_batch_dims(self):
        return list(itertools.chain.from_iterable(
            m.hidden_batch_dims
            for m in self.submodels
        ))
