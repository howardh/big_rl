import torch

from big_rl.model.input_module.modules import _scalar_to_log_unary
from big_rl.model.dense import make_ff_model


class LSTMModel1(torch.nn.Module):
    def __init__(self, input_size: int, action_size: int, hidden_size: int = 128, num_layers: int = 1, ff_size: list[int] = [], unary_energy: bool = False):
        super().__init__()

        self._input_size = input_size
        self._action_size = action_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._unary_energy = unary_energy

        self._key_order = None

        if unary_energy:
            self._energy_size = 8
            input_size = input_size + (self._energy_size - 1)

        if len(ff_size) > 0:
            self.ff = make_ff_model(input_size=input_size, output_size=ff_size[-1], hidden_size=ff_size[:-1])
            input_size = ff_size[-1]
        else:
            self.ff = lambda x: x
        self.lstm = torch.nn.ModuleList([
            torch.nn.LSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size
            )
            for i in range(num_layers)
        ])
        self.fc_value = torch.nn.Linear(hidden_size, 1)
        self.fc_action = torch.nn.Linear(hidden_size, action_size * 2)

    def forward(self, x, hidden):
        if self._key_order is None:
            self._key_order = sorted(x.keys())
        hidden = [h.squeeze(0) for h in hidden]
        prev_h = hidden[:self._num_layers]
        prev_c = hidden[self._num_layers:]

        if self._unary_energy:
            x = {**x} # shallow copy
            x['energy'] = _scalar_to_log_unary(x['energy'], 8)

        x = torch.cat([x[key] for key in self._key_order], dim=-1)
        x = self.ff(x)

        new_h = []
        new_c = []
        for lstm, (h,c) in zip(self.lstm, zip(prev_h, prev_c)):
            h, c = lstm(x, (h,c))
            new_h.append(h.unsqueeze(0))
            new_c.append(c.unsqueeze(0))
            x = h

        value = self.fc_value(x)
        action = self.fc_action(x)
        action_mean = action[:, :self._action_size]
        action_logstd = action[:, self._action_size:]

        return {
            'hidden': (*new_h, *new_c),
            'value': value,
            'action_mean': action_mean,
            'action_logstd': action_logstd,
            'misc': {},
        }

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return tuple(
            torch.zeros(1, batch_size, self._hidden_size, device=device)
            for _ in range(self._num_layers * 2)
        )

    @property
    def n_hidden(self):
        return 2 * self._num_layers

