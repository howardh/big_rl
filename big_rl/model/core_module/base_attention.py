"""
Base core model for attention-based modules.
"""

from __future__ import annotations
from typing import Tuple

import torch
from torchtyping.tensor_type import TensorType

from big_rl.model.core_module.container import CoreModule, CoreModuleOutput


class BaseAttentionCoreModule(CoreModule):
    def __init__(self, key_size, value_size, num_heads):
        super().__init__()

        # Preprocess and validate parameters
        if key_size != value_size:
            raise ValueError('Key size must equal value size')

        # Save parameters
        self._input_size = key_size
        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads

        # Model
        self.query = torch.nn.Parameter(torch.randn(1, 1, key_size))
        self.attention = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)

    def forward(self,
            key: TensorType['seq_len','batch_size','key_size',float],
            value: TensorType['seq_len','batch_size','value_size',float],
            hidden: Tuple[torch.Tensor],
        ) -> CoreModuleOutput:
        batch_size = key.shape[1]

        # attn_output: (num_blocks, 1, batch_size, value_size)
        # attn_output_weights: (num_blocks, batch_size, 1, seq_len)
        # The extra size 1 dimension is the number of queries. We only provide 1 query per module, so it's size 1.
        attn_output, attn_output_weights = self.attention(
                query=self.query.expand(1, batch_size, self._key_size),
                key=key,
                value=value,
        )

        # Remove the extra query dimension
        attn_output = attn_output.squeeze(0) # (batch_size, value_size)
        attn_output_weights = attn_output_weights # (batch_size, num_modules, seq_len)

        # Compute output
        output = self.compute_output(attn_output, hidden)

        return { # seq_len = number of inputs receives
            'key': output['key'],
            'value': output['value'],
            'hidden': output['hidden'],
            'misc': {
                'attn_output': attn_output,
                'attn_output_weights': attn_output_weights,
                **output.get('misc', {}),
            },
        }

    def compute_output(self, attn_output: torch.Tensor, hidden: Tuple[torch.Tensor, ...]) -> dict:
        raise NotImplementedError()

    def init_hidden(self, batch_size) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError()

    @property
    def n_hidden(self):
        raise NotImplementedError()


