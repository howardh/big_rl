from __future__ import annotations
from typing import Tuple

import torch
from torchtyping.tensor_type import TensorType

from big_rl.model.core_module.container import CoreModule, CoreModuleOutput


class ClockCoreModule(CoreModule):
    def __init__(self, key_size, value_size, num_heads, min_init_period=5, max_init_period=1000):
        super().__init__()

        # Model
        self.period = torch.nn.Parameter(torch.rand(1, 1, key_size) * (max_init_period - min_init_period) + min_init_period)
        self.amplitude = torch.nn.Parameter(torch.randn(1, 1, key_size))
        self.phase = torch.nn.Parameter(torch.rand(1, 1, key_size) * self.period)

        self.key = torch.nn.Parameter(torch.randn(1, 1, key_size))

    def forward(self,
            key: TensorType['seq_len','batch_size','key_size',float],
            value: TensorType['seq_len','batch_size','value_size',float],
            hidden: Tuple[torch.Tensor],
        ) -> CoreModuleOutput:

        t = hidden[0]
        batch_size = t.shape[1]

        # Compute the clock
        clock = self.amplitude * torch.sin(2 * torch.pi * t / self.period + self.phase)

        return { # seq_len = number of inputs receives
            'key': self.key.expand(1, batch_size, -1),
            'value': clock,
            'hidden': (t + 1,),
            'misc': {
                #'attn_output': None,
                #'attn_output_weights': None,
                'output_labels': ['clock'],
            }
        }

    def init_hidden(self, batch_size) -> Tuple[torch.Tensor, ...]:
        device = next(self.parameters()).device
        # Start the clock at random phase
        return (
            torch.randint(0, int(self.period.abs().max() * 2 + 1), (batch_size,), device=device).view(1, batch_size, 1),
        )

    @property
    def n_hidden(self):
        return 1
