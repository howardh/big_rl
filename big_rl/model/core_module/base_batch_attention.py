from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from typing import Tuple, List, Type

import torch
from torchtyping.tensor_type import TensorType
from big_rl.model.core_module.base_attention import BaseAttentionCoreModule
from big_rl.model.core_module.container import CoreModule

from big_rl.model.model import BatchMultiHeadAttentionEinsum, NonBatchMultiHeadAttention, BatchMultiHeadAttentionBroadcast
from big_rl.model.model import BatchLinear


class BaseBatchAttentionCoreModule(CoreModule):
    """
    A base model whose multihead attention block uses a fixed query (i.e. independent of the input) to 
    """
    def __init__(self, key_size: int, value_size: int, num_heads: int, num_modules: int = 1, batch_type: str = 'einsum'):
        torch.nn.Module.__init__(self)

        # Preprocess and validate parameters
        if key_size != value_size:
            raise ValueError('Key size must equal value size')

        # Save parameters
        self._input_size = key_size
        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads
        self._num_modules = num_modules

        # Model
        self.query = torch.nn.Parameter(torch.randn(num_modules, 1, key_size))

        # Initialize attention module
        batch_type_mapping = {
            'einsum': BatchMultiHeadAttentionEinsum,
            'none': NonBatchMultiHeadAttention,
            'broadcast': BatchMultiHeadAttentionBroadcast,
        }
        if batch_type not in batch_type_mapping:
            raise ValueError(f"Unknown batch_type {batch_type}. Valid values are: {', '.join(batch_type_mapping.keys())}.")
        MhaClass = batch_type_mapping[batch_type]
        self.attention = MhaClass([
            torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
            for _ in range(num_modules)
        ], key_size=key_size, num_heads=num_heads, default_batch=True)

    def forward(self,
            key: TensorType['seq_len','batch_size','key_size',float],
            value: TensorType['seq_len','batch_size','value_size',float],
            hidden: Tuple,
        ):
        num_modules = self._num_modules
        batch_size = key.shape[1]

        # attn_output: (num_blocks, 1, batch_size, value_size)
        # attn_output_weights: (num_blocks, batch_size, 1, seq_len)
        # The extra size 1 dimension is the number of queries. We only provide 1 query per module, so it's size 1.
        attn_output, attn_output_weights = self.attention(
                self.query.expand([num_modules, *key.shape[1:]]),
                key.expand([num_modules, *key.shape]),
                value.expand([num_modules, *value.shape])
        )

        # Remove the extra query dimension
        attn_output = attn_output.squeeze(1) # (num_blocks, batch_size, value_size)
        attn_output_weights = attn_output_weights.squeeze(2) # (num_blocks, batch_size, seq_len)

        output = self.compute_output(attn_output, hidden)

        return { # seq_len = number of inputs receives
            'key': output['key'], # (num_blocks, batch_size, key_size)
            'value': output['value'], # (num_blocks, batch_size, value_size)
            'hidden': output['hidden'],
            'misc': {
                'attn_output': attn_output, # (num_blocks, batch_size, value_size)
                'attn_output_weights': attn_output_weights, # (num_blocks, batch_size, seq_len)
                **output.get('misc', {}),
            },
        }

    def compute_output(self, attn_output: torch.Tensor, hidden: Tuple[torch.Tensor, ...]) -> dict:
        raise NotImplementedError()

    def _make_mlp(self, sizes, num_duplicates=1, start_with_nonlinearity=True):
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.ReLU())
            layers.append(
                BatchLinear([
                    torch.nn.Linear(in_size, out_size) for _ in range(self._num_modules*num_duplicates)
                ], default_batch=True),
            )
        if not start_with_nonlinearity:
            layers.pop(0) # Remove the first ReLU
        return torch.nn.Sequential(*layers)

    def init_hidden(self, batch_size) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError()

    @property
    def n_hidden(self):
        raise NotImplementedError()

    def to_nonbatched(self) -> BaseAttentionCoreModule:
        raise NotImplementedError()

    @classmethod
    def from_nonbatched(cls, obj) -> BaseBatchAttentionCoreModule:
        raise NotImplementedError()
