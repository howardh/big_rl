import itertools

import torch

from .container import CoreModule


class GatedContainer(CoreModule):
    def __init__(self, key_size, value_size, num_heads, modules: list[CoreModule] = []):
        super().__init__()

        # Save parameters
        self._input_size = key_size
        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads

        self.submodules = torch.nn.ModuleList(modules)

        # Model
        self.query = torch.nn.Parameter(torch.randn(1, 1, key_size))
        self.attn = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
        self.ff_gate = torch.nn.Linear(value_size, len(modules))

    def forward(self, key, value, hidden):
        batch_size = key.shape[1]

        if len(self.submodules) == 0:
            return {
                'key': torch.empty(0, batch_size, self._key_size),
                'value': torch.empty(0, batch_size, self._value_size),
                'hidden': tuple(),
                'misc': {},
            }

        hidden_iter = iter(hidden)
        split_hidden = [
                list(itertools.islice(hidden_iter, m.n_hidden))
                for m in self.submodules
        ]

        module_outputs = [
            m(key, value, h)
            for m,h in zip(self.submodules, split_hidden)
        ]

        new_hidden = tuple(itertools.chain.from_iterable(
            m['hidden']
            for m in module_outputs
        ))

        attn_output, attn_output_weights = self.attn(
                query=self.query.expand(1, batch_size, self._key_size),
                key=key,
                value=value,
        )

        attn_output = attn_output.squeeze(0)
        attn_output_weights = attn_output_weights.squeeze(1)

        gate = torch.sigmoid(self.ff_gate(attn_output))

        # key shape from the submodules is [num_blocks, batch_size, key_size]
        # Stack them to get [num_submodules, num_blocks, batch_size, key_size]
        # Value is the same
        # Gate is [batch_size, num_submodules]. Reshape that to [num_submodules, 1, batch_size, 1] so we can broadcast it
        gate = gate.permute(1,0).unsqueeze(1).unsqueeze(-1)
        key = (gate * torch.stack([m['key'] for m in module_outputs], dim=0)).sum(dim=0)
        value = (gate * torch.stack([m['value'] for m in module_outputs], dim=0)).sum(dim=0)

        return {
            'key': key,
            'value': value,
            'hidden': new_hidden,
            'misc': {
                'attn_output': attn_output,
                'attn_output_weights': attn_output_weights,
            },
        }

    def init_hidden(self, batch_size) -> tuple[torch.Tensor, ...]:
        return tuple(itertools.chain.from_iterable([m.init_hidden(batch_size) for m in self.submodules]))

    @property
    def n_hidden(self):
        return sum(m.n_hidden for m in self.submodules)

    @property
    def hidden_batch_dims(self):
        return list(itertools.chain.from_iterable(
            m.hidden_batch_dims
            for m in self.submodules
        ))
