import itertools
from typing import Any, TypedDict

import torch

from big_rl.model.input_module.modules import _scalar_to_log_unary
from big_rl.model.dense import make_ff_model
from .lstm_model_4 import LSTMModel4
from .ff_model import FFModel


class SubmodelConfig(TypedDict):
    name: str
    kwargs: dict[str, Any]


def make_submodel(name: str, input_size: int, action_size: int, ff_size: list[int]):
    if name == 'LSTMModel4':
        return LSTMModel4(input_size, action_size, ff_size)
    elif name == 'FFModel':
        return FFModel(input_size, action_size, ff_size)
    else:
        raise ValueError(f'Unknown submodel name: {name}')


class LSTMGateModel1(torch.nn.Module):
    """ A model which holds multiple LSTM models and selects between them. Outputs a weighted average of the two models' outputs.
    
    Supported submodels:
    - LSTMModel4
    - FFModel
    """
    def __init__(self, input_size: int, action_size: int, ff_size: list[int] = [], submodel_configs=[]):
        super().__init__()

        self._input_size = input_size
        self._action_size = action_size
        self._ff_size = ff_size
        self._hidden_size = ff_size[-1]

        self.submodels = torch.nn.ModuleList([
            make_submodel(config['name'], input_size, action_size, **config['kwargs'])
            for config in submodel_configs
        ])

        if len(ff_size) < 2:
            raise ValueError(f'LSTMGateModel1 requires at least 2 hidden layers. Only {len(ff_size)} layer(s) were specified via the `ff_size` parameter ({ff_size}).')

        self.ff = make_ff_model(input_size=input_size, output_size=ff_size[-1], hidden_size=ff_size[:-2])
        self.lstm = torch.nn.LSTMCell(ff_size[-2], ff_size[-1])
        output_size = ff_size[-1]
        self.fc_gate = torch.nn.Linear(output_size, len(submodel_configs))

    def forward(self, x, hidden):
        x = self.ff(x)
        h, c = self.lstm(x, (hidden[0], hidden[1]))
        x = h
        gate = self.fc_gate(x)
        gate = gate.softmax(dim=-1)

        hidden_iter = iter(hidden[2:])
        split_hiddens = [
            itertools.islice(hidden_iter, m.n_hidden)
            for m in self.submodels
        ]
        submodel_outputs = [
            m(x, tuple(h))
            for m, h in zip(self.submodels, split_hiddens)
        ]

        value = sum(gate[:, i:i+1] * o['value'] for i, o in enumerate(submodel_outputs))
        action_mean = sum(gate[:, i:i+1] * o['action_mean'] for i, o in enumerate(submodel_outputs))
        action_logstd = sum(gate[:, i:i+1] * o['action_logstd'] for i, o in enumerate(submodel_outputs))

        new_hidden = (h, c) + tuple(itertools.chain.from_iterable(o['hidden'] for o in submodel_outputs))

        return {
            'hidden': new_hidden,
            'value': value,
            'action_mean': action_mean,
            'action_logstd': action_logstd,
            'misc': {},
        }

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return tuple(
            [
                torch.zeros(batch_size, self._hidden_size, device=device),
                torch.zeros(batch_size, self._hidden_size, device=device),
            ]+ list(itertools.chain.from_iterable(m.init_hidden(batch_size) for m in self.submodels))
        )

    @property
    def n_hidden(self):
        return 2 + sum(m.n_hidden for m in self.submodels)

    @property
    def hidden_batch_dims(self):
        return [0, 0] + list(itertools.chain.from_iterable(m.hidden_batch_dims for m in self.submodels))

