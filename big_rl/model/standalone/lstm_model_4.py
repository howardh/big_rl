import torch

from big_rl.model.input_module.modules import _scalar_to_log_unary
from big_rl.model.dense import make_ff_model


class LSTMModel4(torch.nn.Module):
    """ A plain fully connected model with a LSTM layer at the end. """
    def __init__(self, input_size: int, action_size: int, ff_size: list[int] = []):
        super().__init__()

        self._input_size = input_size
        self._action_size = action_size
        self._ff_size = ff_size
        self._hidden_size = ff_size[-1]

        if len(ff_size) < 2:
            raise ValueError(f'LSTMModel4 requires at least 2 hidden layers. Only {len(ff_size)} layer(s) were specified via the `ff_size` parameter ({ff_size}).')

        self.ff = make_ff_model(input_size=input_size, output_size=ff_size[-1], hidden_size=ff_size[:-2])
        self.lstm = torch.nn.LSTMCell(ff_size[-2], ff_size[-1])
        output_size = ff_size[-1]
        self.fc_value = torch.nn.Linear(output_size, 1)
        self.fc_action = torch.nn.Linear(output_size, action_size * 2)

    def forward(self, x, hidden):
        x = self.ff(x)
        h, c = self.lstm(x, hidden)
        x = h

        value = self.fc_value(x)
        action = self.fc_action(x)
        action_mean = action[:, :self._action_size]
        action_logstd = action[:, self._action_size:]

        return {
            'hidden': (h, c),
            'value': value,
            'action_mean': action_mean,
            'action_logstd': action_logstd,
            'misc': {},
        }

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self._hidden_size, device=device),
            torch.zeros(batch_size, self._hidden_size, device=device),
        )

    @property
    def n_hidden(self):
        return 2

    @property
    def hidden_batch_dims(self):
        return [0, 0]

    def decapitate(self, *args, **kwargs):
        return LSTMModel4Headless(self, *args, **kwargs)

class LSTMModel4Headless(torch.nn.Module):
    def __init__(self, model: LSTMModel4, remove_input_head: bool = True):
        super().__init__()

        self._model = model

        if remove_input_head:
            if len(self._model.ff) < 2:
                raise ValueError(f'LSTMModel4 requires at least 2 hidden layers if the input head is to be removed. Only {len(self._model.ff)} layer(s) were specified.')
            self.ff = self._model.ff[1:] # XXX: Is this correct?
        else:
            self.ff = self._model.ff

    def forward(self, x, hidden):
        x = self.ff(x)
        h, c = self._model.lstm(x, hidden)
        x = h

        return {
            'hidden': (h, c),
            'output': x,
            'misc': {},
        }

    def init_hidden(self, batch_size):
        return self._model.init_hidden(batch_size)

    @property
    def n_hidden(self):
        return self._model.n_hidden

    @property
    def hidden_batch_dims(self):
        return self._model.hidden_batch_dims

    @property
    def input_size(self):
        return self._model.ff[0].in_features

    @property
    def output_size(self):
        return self._model._hidden_size
