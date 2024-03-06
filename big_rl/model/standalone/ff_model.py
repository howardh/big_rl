import torch

from big_rl.model.input_module.modules import _scalar_to_log_unary
from big_rl.model.dense import make_ff_model


class FFModel(torch.nn.Module):
    """ A plain fully connected feed-forward model for continuous action spaces. """
    def __init__(self, input_size: int, action_size: int, ff_size: list[int] = []):
        super().__init__()

        self._input_size = input_size
        self._action_size = action_size
        self._ff_size = ff_size

        self.ff = make_ff_model(input_size=input_size, output_size=ff_size[-1], hidden_size=ff_size[:-1])
        output_size = ff_size[-1]
        self.fc_value = torch.nn.Linear(output_size, 1)
        self.fc_action = torch.nn.Linear(output_size, action_size * 2)

    def forward(self, x, hidden):
        x = self.ff(x)
        value = self.fc_value(x)
        action = self.fc_action(x)
        action_mean = action[:, :self._action_size]
        action_logstd = action[:, self._action_size:]

        return {
            'hidden': tuple(),
            'value': value,
            'action_mean': action_mean,
            'action_logstd': action_logstd,
            'misc': {},
        }

    def init_hidden(self, batch_size):
        return tuple()

    @property
    def n_hidden(self):
        return 0

    @property
    def hidden_batch_dims(self):
        return []

