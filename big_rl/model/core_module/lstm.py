"""
LSTM core module compatible with CoreModuleContainer API

WIP
"""

from __future__ import annotations
from typing import Tuple

import torch
from torchtyping.tensor_type import TensorType

from big_rl.model.core_module.container import CoreModule, CoreModuleOutput
from big_rl.model.attention_input import AttentionInput


class AttentionLSTM(CoreModule):
    def __init__(self, key_size, value_size, num_heads, hidden_size: int = 128, dynamic_query: bool = False):
        super().__init__()

        # Preprocess and validate parameters
        assert key_size == value_size

        # Save parameters
        self._input_size = key_size
        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads
        self._hidden_size = hidden_size
        self._dynamic_query = dynamic_query

        # Model
        self.attention_input = AttentionInput(key_size=key_size, value_size=value_size, num_heads=num_heads, dynamic_query=dynamic_query)
        self.fc = torch.nn.Linear(hidden_size, value_size*2) # For producing the key and value from the hidden state
        self.attention = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
        self.lstm = torch.nn.LSTMCell(
                input_size=self._input_size,
                hidden_size=hidden_size,
        )

    def forward(self,
            key: TensorType['seq_len','batch_size','key_size',float],
            value: TensorType['seq_len','batch_size','value_size',float],
            hidden: Tuple[torch.Tensor, torch.Tensor],
        ) -> CoreModuleOutput:
        batch_size = key.shape[1]

        prev_hidden_state = hidden[0]
        prev_cell_state = hidden[1]

        # attn_output: (1, batch_size, value_size)
        # attn_output_weights: (batch_size, seq_len)
        x = self.attention_input(
                key=key,
                value=value,
        )
        attn_output = x['attn_output']
        attn_output_weights = x['attn_output_weights']

        new_hidden_state, new_cell_state = self.lstm(
                attn_output, 
                (prev_hidden_state, prev_cell_state),
        )

        fc_output = self.fc(new_hidden_state)
        new_key = fc_output[:, :self._key_size]
        new_value = fc_output[:, self._key_size:]

        return { # seq_len = number of inputs receives
            'key': new_key.unsqueeze(0),
            'value': new_value.unsqueeze(0),
            'hidden': (
                new_hidden_state,
                new_cell_state,
            ),
            'misc': {
                'attn_output': attn_output,
                'attn_output_weights': attn_output_weights,
            },
        }

    def init_hidden(self, batch_size) -> Tuple[torch.Tensor, ...]:
        device = next(self.parameters()).device
        return (
            torch.zeros([batch_size, self._hidden_size], device=device),
            torch.zeros([batch_size, self._hidden_size], device=device),
        )

    @property
    def n_hidden(self):
        return 2

    @property
    def hidden_batch_dims(self):
        return [0, 0]
