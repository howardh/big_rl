from .container import CoreModule, CoreModuleParallel, CoreModuleSeries
from .headless_container import HeadlessContainer
from .gated_container import GatedContainer
from .base_attention import BaseAttentionCoreModule
from .base_batch_attention import BaseBatchAttentionCoreModule
from .lstm import AttentionLSTM
from .recurrent_attention_17 import RecurrentAttention17, NonBatchRecurrentAttention17
from .clock import ClockCoreModule
from .generalized_hebbian_algorithm import AttentionGHA


AVAILABLE_CORE_MODULES = [
    cls
    for cls in CoreModule.subclasses
    if cls not in [BaseAttentionCoreModule, BaseBatchAttentionCoreModule]
]


__all__ = [
    'CoreModule',
    'CoreModuleParallel',
    'CoreModuleSeries',

    'HeadlessContainer',
    'GatedContainer',

    'AttentionLSTM',
    'RecurrentAttention17',
    'NonBatchRecurrentAttention17',
    'ClockCoreModule',
    'AttentionGHA',
]
