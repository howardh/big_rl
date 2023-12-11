"""
LSTM core module compatible with CoreModuleContainer API

WIP
"""

from __future__ import annotations
from typing import Tuple

import torch
from torchtyping.tensor_type import TensorType

from big_rl.model.core_module.base_attention import BaseAttentionCoreModule
from big_rl.model.core_module.base_batch_attention import BaseBatchAttentionCoreModule


def add_bias_dim(x, dim=1):
    shape = list(x.shape)
    shape[dim] = 1
    return torch.cat([x, torch.ones(shape, device=x.device)], dim=dim)


def generalized_hebbian_algorithm(x, weight, learning_rate):
    x = x.unsqueeze(2) # (batch_size, input_size, 1)
    x_with_bias = add_bias_dim(x, dim=1) # (batch_size, input_size+1, 1)
    y = weight @ x_with_bias # (batch_size, output_size, 1)
    return weight + learning_rate * (y @ x_with_bias.transpose(1, 2) - torch.tril(y @ y.transpose(1, 2)) @ weight)


batch_generalized_hebbian_algorithm = torch.func.vmap(generalized_hebbian_algorithm, in_dims=(0, 0, 0), out_dims=0)


class AttentionGHA(BaseAttentionCoreModule):
    def __init__(self, key_size, value_size, num_heads, num_components: int = 128, learning_rate: float = 0.01):
        super().__init__(key_size=key_size, value_size=value_size, num_heads=num_heads)

        self._num_components = num_components

        # Model
        self.fc = torch.nn.Linear(num_components, value_size*2) # For producing the key and value from the hidden state
        self.log_learning_rate = torch.nn.Parameter(torch.log(torch.tensor(learning_rate)))

    def compute_output(self, attn_output: torch.Tensor, hidden: Tuple[torch.Tensor, ...]) -> dict:
        prev_weights = hidden[0]

        new_weights = generalized_hebbian_algorithm(attn_output, prev_weights, self.log_learning_rate.exp()) # (batch_size, value_size, value_size)
        y = new_weights @ add_bias_dim(attn_output, dim=1).unsqueeze(2)
        y = y.squeeze(2).abs() # (batch_size, value_size)

        fc_output = self.fc(y)
        new_key = fc_output[:, :self._key_size].unsqueeze(0)
        new_value = fc_output[:, self._key_size:].unsqueeze(0)

        return {
            'key': new_key,
            'value': new_value,
            'hidden': (
                new_weights,
            )
        }

    def init_hidden(self, batch_size) -> Tuple[torch.Tensor, ...]:
        device = next(self.parameters()).device
        return (
            torch.rand([batch_size, self._num_components, self._value_size+1], device=device),
        )

    @property
    def n_hidden(self):
        return 1


class BatchAttentionGHA(BaseBatchAttentionCoreModule):
    def __init__(self, key_size, value_size, num_heads, num_components: int = 128, learning_rate: float = 0.01, num_modules: int = 1, ff_size: list[int] = [], learnable_initial_state: bool = False):
        super().__init__(key_size=key_size, value_size=value_size, num_heads=num_heads, num_modules=num_modules)

        self._num_components = num_components
        self._learnable_initial_state = learnable_initial_state

        # Model
        self.fc = self._make_mlp(
            [num_components, *ff_size, value_size*2],
            start_with_nonlinearity=False
        ) # For producing the key and value
        self.log_learning_rate = torch.nn.Parameter(
            torch.log(torch.ones(num_modules) * learning_rate)
        )
        if learnable_initial_state:
            self.initial_hidden = torch.nn.Parameter(
                torch.rand([self._num_modules, 1, self._num_components, self._value_size+1])
            )

    def compute_output(self, attn_output: torch.Tensor, hidden: Tuple[torch.Tensor, ...]) -> dict:
        prev_weights = hidden[0]

        new_weights = batch_generalized_hebbian_algorithm(attn_output, prev_weights, self.log_learning_rate.exp()) # (num_modules, batch_size, num_components, value_size)
        new_weights = new_weights / new_weights.pow(2).sum(dim=3, keepdim=True).sqrt() # Normalize
        y = new_weights @ add_bias_dim(attn_output, dim=2).unsqueeze(3)
        y = y.squeeze(3).abs() # (num_modules, batch_size, num_components)

        fc_output = self.fc(y).tanh()
        new_key = fc_output[:, :, :self._key_size]
        new_value = fc_output[:, :, self._key_size:]

        if torch.isnan(fc_output).any():
            breakpoint()

        return {
            'key': new_key, # (num_modules, batch_size, key_size)
            'value': new_value,
            'hidden': (
                new_weights,
            ),
            'misc': {
                'output_labels': ['gha'] * self._num_modules
            }
        }

    def init_hidden(self, batch_size) -> Tuple[torch.Tensor, ...]:
        device = next(self.parameters()).device
        if self._learnable_initial_state:
            h = self.initial_hidden.expand(self._num_modules, batch_size, self._num_components, self._value_size+1)
        else:
            h = torch.rand([self._num_modules, batch_size, self._num_components, self._value_size+1], device=device)
        h = h / h.pow(2).sum(dim=3, keepdim=True).sqrt() # Normalize
        return (h,)

    @property
    def n_hidden(self):
        return 1
