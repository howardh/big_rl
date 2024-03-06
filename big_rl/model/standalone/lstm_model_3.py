import torch

from big_rl.model.input_module.modules import _scalar_to_log_unary
from big_rl.model.dense import make_ff_model



batch_linear = torch.func.vmap(
    torch.nn.functional.linear,
    in_dims=0,
    out_dims=0,
)


class LearnableLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, rank=1):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._rank = rank

        self.initial_weight = torch.nn.Parameter(torch.randn(1, output_size, input_size))
        self.initial_bias = torch.nn.Parameter(torch.zeros(1, output_size))

        if rank > 0:
            self._recurrence_output_size = (input_size + output_size) * rank + output_size
        elif rank == -1:
            self._recurrence_output_size = output_size * input_size + output_size
        else:
            raise ValueError('rank must be positive, or -1 for full rank')
        self.lstm = torch.nn.LSTMCell(input_size, self._recurrence_output_size)

    def forward(self, x, hidden):
        batch_size, input_size = x.shape

        assert input_size == self._input_size

        #h, c = self.lstm(x, hidden)
        #batch_lstm = torch.func.vmap(
        #    lambda x, h, c: self.lstm(x,(h,c)),
        #    in_dims=0,
        #    out_dims=0,
        #)
        #h, c = batch_lstm(x, hidden[0], hidden[1])
        h, c = torch.utils.data.default_collate([
            self.lstm(a, (h, c))
            for a,h,c in zip(x, hidden[0], hidden[1])
        ])

        if self._rank == -1:
            dw, db = h.split([
                self._output_size * self._input_size,
                self._output_size
            ], dim=-1)
            dw = dw.view(batch_size, self._output_size, self._input_size)
        else:
            dw_1, dw_2, db = h.split([
                self._output_size * self._rank,
                self._input_size * self._rank,
                self._output_size
            ], dim=-1)
            dw_1 = dw_1.view(batch_size, self._output_size, self._rank)
            dw_2 = dw_2.view(batch_size, self._rank, self._input_size)
            dw = dw_1 @ dw_2
        weight = self.initial_weight + dw
        bias = self.initial_bias + db

        return batch_linear(x, weight, bias), (h, c)

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self._recurrence_output_size, device=device),
            torch.zeros(batch_size, self._recurrence_output_size, device=device),
        )

    @property
    def n_hidden(self):
        return 2

    @property
    def hidden_batch_dims(self):
        return [0, 0]


class LSTMModel3(torch.nn.Module):
    """ Model whose input layer weights are adjusted over the course of an episode via the outputs of a recurrent module. """
    def __init__(self, input_size: int, action_size: int, ff_size: list[int] = [], rank: int = 1):
        raise NotImplementedError('This model is not yet implemented')
        super().__init__()

        self._input_size = input_size
        self._action_size = action_size
        self._ff_size = ff_size

        if len(ff_size) >= 2:
            self.ff_1 = LearnableLinear(input_size, ff_size[0], rank=rank)
            self.ff_2 = make_ff_model(input_size=ff_size[0], output_size=ff_size[-1], hidden_size=ff_size[1:-1])
            output_size = ff_size[-1]
        elif len(ff_size) == 1:
            self.ff_1 = LearnableLinear(input_size, ff_size[0], rank=rank)
            self.ff_2 = lambda x: x
            output_size = ff_size[0]
        else:
            raise ValueError('ff_size must have at least one element')
        self.fc_value = torch.nn.Linear(output_size, 1)
        self.fc_action = torch.nn.Linear(output_size, action_size * 2)

    def forward(self, x, hidden):
        x, new_hidden = self.ff_1(x, hidden)
        x = self.ff_2(x)

        value = self.fc_value(x)
        action = self.fc_action(x)
        action_mean = action[:, :self._action_size]
        action_logstd = action[:, self._action_size:]

        return {
            'hidden': new_hidden,
            'value': value,
            'action_mean': action_mean,
            'action_logstd': action_logstd,
            'misc': {},
        }

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return tuple([
            *self.ff_1.init_hidden(batch_size),
            torch.zeros(batch_size, self._recurrence_output_size, device=device),
        ])

    @property
    def n_hidden(self):
        return self.ff_1.n_hidden

    @property
    def hidden_batch_dims(self):
        return self.ff_1.hidden_batch_dims

