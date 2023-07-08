import inspect
import typing
from typing import TypedDict, TypeAlias, Type
from collections import defaultdict
import itertools
import warnings

import torch
from jaxtyping import Float


KeyInputType: TypeAlias = Float[torch.Tensor, 'num_inputs batch_size key_size']
ValueInputType: TypeAlias = Float[torch.Tensor, 'num_inputs batch_size value_size']
KeyOutputType: TypeAlias = Float[torch.Tensor, 'num_modules batch_size key_size']
ValueOutputType: TypeAlias = Float[torch.Tensor, 'num_modules batch_size value_size']
HiddenType: TypeAlias = tuple[torch.Tensor, ...]


class CoreModuleOutput(TypedDict):
    key: KeyOutputType
    value: ValueOutputType
    hidden: HiddenType


_core_module_subclasses = []


def _validate_core_module_constructor(cls):
    # Make sure the subclass constructor has the same signature as the parent
    # Check it upon subclassing so that it is caught immediately and we can provide a better error message

    # Get the signature of the subclassed constructor
    # We need both `inspect.signature` and `typing.get_type_hints` because the latter omits arguments without type hints.
    constructor_signature = inspect.signature(cls.__init__)
    constructor_type_hints = typing.get_type_hints(cls.__init__)

    # The constructor should have the following arguments:
    # - key_size: int
    # - value_size: int
    # - num_heads: int
    required_args = [
        ('key_size', int),
        ('value_size', int),
        ('num_heads', int),
    ]
    for arg_name, arg_type in required_args:
        if arg_name not in constructor_signature.parameters:
            raise TypeError(f'Expected {cls.__name__} to have an argument named {arg_name} of type {arg_type.__name__}')
        if constructor_signature.parameters[arg_name].annotation == inspect.Parameter.empty:
            warnings.warn(f'Expected {cls.__name__} to have an argument named {arg_name} of type {arg_type.__name__}. Found argument named {arg_name}, but type is not specified.')
        elif constructor_type_hints[arg_name] != arg_type:
            raise TypeError(f'Expected {cls.__name__} to have an argument named {arg_name} of type {arg_type.__name__}. Found type {constructor_signature.parameters[arg_name].annotation}')

    # All other arguments must have default values
    for arg_name, arg in constructor_signature.parameters.items():
        if arg_name == 'self':
            continue
        if arg_name in [arg_name for arg_name, _ in required_args]:
            continue
        if arg.default == inspect.Parameter.empty:
            raise TypeError(f'Expected {cls.__name__} to have a default value for argument `{arg_name}`.')


def _validate_core_module_forward(cls):
    # Make sure the subclass constructor has the same signature as the parent
    # Check it upon subclassing so that it is caught immediately and we can provide a better error message

    # Get the signature of the subclassed constructor
    # We need both `inspect.signature` and `typing.get_type_hints` because the latter omits arguments without type hints.
    signature = inspect.signature(cls.forward)

    # The forward method should have the following arguments:
    # - key: torch.Tensor
    # - value: torch.Tensor
    # - hidden: tuple[torch.Tensor, ...]
    required_args = ['key', 'value', 'hidden']
    for arg_name in required_args:
        if arg_name not in signature.parameters:
            raise TypeError(f'Expected {cls.__name__}.forward() to have an argument named {arg_name}.')

    # Check argument order. It should be (key, value, hidden)
    if list(signature.parameters.keys())[1:4] != required_args:
        raise TypeError(f'Expected {cls.__name__}.forward() to have arguments in the following order: {required_args}. Found {list(signature.parameters.keys())[1:4]}')


class CoreModule(torch.nn.Module):
    @classmethod
    @property
    def subclasses(cls):
        return _core_module_subclasses

    def __init_subclass__(cls, **kwargs):
        # If it's a container or a subclass of a container, then do nothing
        # Note: `CoreModuleContainer` has to be defined immediately after `CoreModule` in order for this to work. When `CoreModuleContainer` is defined, only the first condition is evaluated. The second one, which would error because `CoreModuleContainer` is not defined, is skipped.
        if cls.__name__ == 'CoreModuleContainer' or issubclass(cls, CoreModuleContainer):
            return

        # If it is an abstract class, then do nothing
        if inspect.isabstract(cls):
            return

        _validate_core_module_constructor(cls)
        _validate_core_module_forward(cls)

        # Add the subclass to the list of subclasses
        _core_module_subclasses.append(cls)

    def __call__(self, key: KeyInputType, value: ValueInputType, hidden: HiddenType) -> CoreModuleOutput:
        if __debug__:
            if len(key.shape) != 3:
                raise ValueError(f'Expected key to have shape (batch_size, num_inputs, key_size), got {key.shape}')
            if len(value.shape) != 3:
                raise ValueError(f'Expected value to have shape (batch_size, num_inputs, value_size), got {value.shape}')

        output = super().__call__(key, value, hidden)

        if __debug__:
            _, batch_size, key_size = key.shape
            value_size = value.shape[2]
            if 'key' not in output:
                raise ValueError(f'Core module output missing "key" field')
            if 'value' not in output:
                raise ValueError(f'Core module output missing "value" field')
            if 'hidden' not in output:
                raise ValueError(f'Core module output missing "hidden" field. If there is no hidden state, return an empty tuple.')
            if len(output['key'].shape) != 3:
                raise ValueError(f'Expected key to have shape (batch_size, num_modules, key_size), got {output["key"].shape}')
            if len(output['value'].shape) != 3:
                raise ValueError(f'Expected value to have shape (batch_size, num_modules, value_size), got {output["value"].shape}')
            if output['key'].shape[1] != batch_size:
                raise ValueError(f'Output key dimensions does not match input key dimension. Expected key to have batch size {batch_size}, got {output["key"].shape[1]}')
            if output['key'].shape[2] != key_size:
                raise ValueError(f'Output key dimensions does not match number of core modules. Expected key to have size {key_size}, got {output["key"].shape[2]}')
            if output['value'].shape[1] != batch_size:
                raise ValueError(f'Output value dimensions does not match input value dimension. Expected value to have batch size {batch_size}, got {output["value"].shape[1]}')
            if output['value'].shape[2] != value_size:
                raise ValueError(f'Output value dimensions does not match input value dimension. Expected value to have size {value_size}, got {output["value"].shape[2]}')

        return output

    def init_hidden(self, batch_size):
        raise NotImplementedError()

    @property
    def n_hidden(self):
        raise NotImplementedError()


class CoreModuleContainer(CoreModule):
    def __init__(self, modules: list[CoreModule]):
        """
        Args:
        """
        super().__init__()

        self.core_modules = modules

        for m in modules:
            if not hasattr(m, 'init_hidden'):
                raise ValueError(f'Core module {m} does not have `init_hidden()` method')
            if not hasattr(m, 'n_hidden'):
                raise ValueError(f'Core module {m} does not have `n_hidden` property')

    def forward(self, key: KeyInputType, value: ValueInputType, hidden: HiddenType) -> CoreModuleOutput:
        raise NotImplementedError()

    def _split_hidden(self, hidden):
        """ Split hidden state into a list of hidden states, one for each core module. """
        output = []
        start_idx = 0
        for m in self.core_modules:
            n = int(m.n_hidden) # type: ignore
            output.append(hidden[start_idx:start_idx+n])
            start_idx += n
        return output

    def init_hidden(self, batch_size) -> tuple[torch.Tensor, ...]:
        return tuple(itertools.chain.from_iterable([m.init_hidden(batch_size) for m in self.core_modules])) # type: ignore

    @property
    def n_hidden(self) -> int:
        if self.core_modules is None:
            return 0
        return sum(m.n_hidden for m in self.core_modules) # type: ignore


class CoreModuleParallel(CoreModuleContainer):
    def __init__(self, modules: list[CoreModule]):
        super().__init__(modules)

    def forward(self, key: KeyInputType, value: ValueInputType, hidden: HiddenType) -> CoreModuleOutput:
        split_hidden = self._split_hidden(hidden)

        new_key = []
        new_value = []
        new_hidden = []
        for m,h in zip(self.core_modules, split_hidden):
            output = m(key, value, h)
            new_key.append(output['key'])
            new_value.append(output['value'])
            new_hidden.append(output['hidden'])

        return {
            'key': torch.cat(new_key, dim=0),
            'value': torch.cat(new_value, dim=0),
            'hidden': tuple(itertools.chain.from_iterable(new_hidden)),
        }


class CoreModuleSeries(CoreModuleContainer):
    def __init__(self, modules: list[CoreModule]):
        super().__init__(modules)

    def forward(self, key: KeyInputType, value: ValueInputType, hidden: HiddenType) -> CoreModuleOutput:
        split_hidden = self._split_hidden(hidden)

        new_key = key
        new_value = value
        new_hidden = []
        for m,h in zip(self.core_modules, split_hidden):
            output = m(new_key, new_value, h)
            new_key = output['key']
            new_value = output['value']
            new_hidden.append(output['hidden'])

        return {
            'key': new_key,
            'value': new_value,
            'hidden': tuple(itertools.chain.from_iterable(new_hidden)),
        }
