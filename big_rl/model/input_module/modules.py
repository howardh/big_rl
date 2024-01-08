from typing import Sequence

import torch
from jaxtyping import Float, Int

from .base import InputModule


class IgnoredInput(InputModule):
    def __init__(self, key_size: int, value_size: int):
        super().__init__()


class GreyscaleImageInput(InputModule):
    def __init__(self, key_size: int, value_size: int, in_channels: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,out_channels=32,kernel_size=8,stride=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.Flatten(),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.LeakyReLU(),
        )
        self.fc_key = torch.nn.Linear(in_features=512, out_features=key_size)
        self.fc_value = torch.nn.Linear(in_features=512, out_features=value_size)
    def forward(self, x: Float[torch.Tensor, 'batch_size frame_stack height width']):
        x = self.conv(x * self.scale)
        return {
            'key': self.fc_key(x),
            'value': self.fc_value(x),
        }


class ImageInput56(InputModule):
    def __init__(self, key_size: int, value_size: int, in_channels: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,out_channels=32,kernel_size=8,stride=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=4,stride=1),
            torch.nn.Flatten(),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.LeakyReLU(),
        )
        self.fc_key = torch.nn.Linear(in_features=512, out_features=key_size)
        self.fc_value = torch.nn.Linear(in_features=512, out_features=value_size)
    def forward(self, x: Float[torch.Tensor, 'batch_size channels height width']):
        x = self.conv(x.float() * self.scale)
        return {
            'key': self.fc_key(x),
            'value': self.fc_value(x),
        }


class ImageInput84(InputModule):
    def __init__(self, key_size: int, value_size: int, in_channels: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,out_channels=32,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.ReLU(),
        )
        self.fc_key = torch.nn.Linear(in_features=512, out_features=key_size)
        self.fc_value = torch.nn.Linear(in_features=512, out_features=value_size)
    def forward(self, x: Float[torch.Tensor, 'batch_size channels height width']):
        x = self.conv(x.float() * self.scale)
        return {
            'key': self.fc_key(x),
            'value': self.fc_value(x),
        }


class ScalarInput(InputModule):
    def __init__(self, key_size: int, value_size: int):
        super().__init__()
        self.value_size = value_size
        self.key = torch.nn.Parameter(torch.rand([key_size]))
    def forward(self, value: Float[torch.Tensor, 'batch_size']):
        batch_size = int(torch.tensor(value.shape).prod().item())
        batch_shape = value.shape
        assert len(batch_shape) == 2
        assert batch_shape[-1] == 1, 'Last dimension of input to ScalarInput has to be size 1.'
        return {
            'key': self.key.view(1,-1).expand(batch_size, -1).view(*batch_shape[:-1],-1),
            'value': value.view(-1,1).expand(batch_size,self.value_size).view(*batch_shape[:-1],-1)
        }


def _scalar_to_log_unary(value: Float[torch.Tensor, 'batch_size'], output_size: int):
    """ Convert a scalar to a "unary" tensor. """
    batch_size = value.shape[0]

    unary_output = torch.zeros([batch_size, output_size], device=value.device)
    log_value = torch.log(value + 1)
    for i in range(batch_size):
        lv = int(log_value[i].item())
        rem = log_value[i].item() - lv
        unary_output[i,:lv] = 1
        unary_output[i,lv] = rem
    return unary_output


class UnaryScalarInput(InputModule):
    def __init__(self, key_size: int, value_size: int, min_output_value: float = -1, max_output_value: float = 1, min_input_value: float = 1e-5, max_input_value: float = float('inf')):
        super().__init__()
        self.value_size = value_size
        self.key = torch.nn.Parameter(torch.rand([key_size]))
        self.min_output_value = min_output_value
        self.max_output_value = max_output_value
    def forward(self, value: Float[torch.Tensor, 'batch_size']):
        batch_size = int(torch.tensor(value.shape).prod().item())
        batch_shape = value.shape
        assert len(batch_shape) == 2
        assert batch_shape[-1] == 1, 'Last dimension of input to UnaryScalarInput has to be size 1.'
        assert (value >= 0).all(), 'UnaryScalarInput only supports positive values.'

        unary_output = _scalar_to_log_unary(value, self.value_size) * (self.max_output_value - self.min_output_value) + self.min_output_value

        return {
            'key': self.key.view(1,-1).expand(batch_size, -1).view(*batch_shape[:-1],-1),
            'value': unary_output
        }


class LinearInput(InputModule):
    def __init__(self, input_size: int, key_size: int, value_size: int, shared_key: bool = False):
        """
        Args:
            input_size: The size of the input vector.
            key_size: The size of the key.
            value_size: The size of the value.
            shared_key: If set to True, the same key will be used regardless of input. If set to False, the key will be computed as a linear function of the input.
        """
        super().__init__()
        self._input_size = input_size
        self._shared_key = shared_key
        self.ff_value = torch.nn.Linear(in_features=input_size, out_features=value_size)
        if shared_key:
            self.key = torch.nn.Parameter(torch.rand([key_size])-0.5)
        else:
            self.ff_key = torch.nn.Linear(in_features=input_size, out_features=key_size)
    def forward(self, x: Float[torch.Tensor, 'batch_size input_size']):
        batch_size = x.shape[0]
        input_size = x.shape[1]
        if input_size != self._input_size:
            raise ValueError(f'Expected input of size {self._input_size} but got {input_size}.')
        return {
            'key': self.ff_key(x) if not self._shared_key else self.key.expand(batch_size, -1),
            'value': self.ff_value(x)
        }


class DiscreteInput(InputModule):
    def __init__(self, input_size: int, key_size: int, value_size: int, shared_key: bool = False):
        """
        Args:
            input_size: Number of possible input values.
            key_size: The size of the key.
            value_size: The size of the value.
            shared_key: If set to True, the same key will be used regardless of input. If set to False, the key will be computed as a linear function of the input.
        """
        super().__init__()
        self._shared_key = shared_key
        self._input_size = input_size
        self.value = torch.nn.Parameter(torch.rand([input_size, value_size])-0.5)
        if shared_key:
            self.key = torch.nn.Parameter(torch.rand([key_size])-0.5)
        else:
            self.key = torch.nn.Parameter(torch.rand([input_size, key_size])-0.5)
    def forward(self, x: Int[torch.Tensor, 'batch_size']):
        batch_size = int(torch.tensor(x.shape).prod().item())
        batch_shape = x.shape
        if len(batch_shape) != 1:
            raise ValueError('Input to DiscreteInput has to be a 1D tensor.')
        if x.min() < 0 or x.max() >= self._input_size:
            raise ValueError(f'Input to DiscreteInput has to be in range [0, {self._input_size}).')
        x = x.long().flatten()
        return {
            'key': self.key.expand(batch_size, -1).view(*batch_shape, -1) if self._shared_key else self.key[x,:].view(*batch_shape, -1),
            'value': self.value[x,:].view(*batch_shape, -1)
        }


class MatrixInput(InputModule):
    def __init__(self, input_size: Sequence[int], key_size: int, value_size: int, num_heads: int = 1, shared_key: bool = False):
        """
        Args:
            input_size: The size of the input matrix.
            key_size: The size of the key.
            value_size: The size of the value.
            num_heads: ...
            shared_key: If set to True, the same key will be used regardless of input. If set to False, the key will be computed as a linear function of the input.
        """
        super().__init__()
        self._shared_key = shared_key
        self._key_size = key_size
        self._value_size = value_size

        self.left_value = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty([num_heads, input_size[0]])
            )
        )
        self.right_value = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty([input_size[1], value_size // num_heads])
            )
        )

        if shared_key:
            self.key = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(
                    torch.empty(key_size)
                )
            )
        else:
            self.left_key = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(
                    torch.empty([num_heads, input_size[0]])
                )
            )
            self.right_key = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(
                    torch.empty([input_size[1], key_size // num_heads])
                )
            )
    def forward(self, x: Float[torch.Tensor, 'batch_size dim1 dim2']):
        batch_size = x.shape[0]
        if self._shared_key:
            key = self.key.expand(batch_size, -1)
        else:
            key = self.left_key @ x @ self.right_key
            key = key.view(batch_size, self._key_size)
        return {
            'key': key,
            'value': (self.left_value @ x @ self.right_value).view(batch_size, self._value_size)
        }

