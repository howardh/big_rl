import pytest
import torch

from big_rl.model.model import BatchLinear, NonBatchLinear
from big_rl.model.model import BatchLinearCpp


@pytest.mark.parametrize("batch_linear_cls", [BatchLinear, BatchLinearCpp])
def test_same_as_non_batch_with_nonbatched_inputs(batch_linear_cls):
    """ If the batched and non-batched versions of the same model are given the same input, they should produce the same output. The inputs are not batched (`default_batch=False`), meaning that the same input is fed to each linear module in the batch of linear modules. """
    num_modules = 5

    modules = [torch.nn.Linear(3, 4) for _ in range(num_modules)]
    batch_linear = batch_linear_cls(modules, default_batch=False)
    non_batch_linear = NonBatchLinear(modules, default_batch=False)

    x = torch.randn(num_modules, 3)
    assert torch.allclose(batch_linear(x), non_batch_linear(x))


@pytest.mark.parametrize("batch_linear_cls", [BatchLinear, BatchLinearCpp])
def test_same_as_non_batch_with_batched_inputs(batch_linear_cls):
    """ If the batched and non-batched versions of the same model are given the same input, they should produce the same output. The inputs are batched (`default_batch=True`), meaning that we have the same number of inputs as linear modules, each of which are fed to exactly one of the linear modules. """
    num_modules = 5

    modules = [torch.nn.Linear(3, 4) for _ in range(num_modules)]
    batch_linear = batch_linear_cls(modules, default_batch=True)
    non_batch_linear = NonBatchLinear(modules, default_batch=True)

    x = torch.randn(num_modules, 4, 3)
    assert torch.allclose(batch_linear(x), non_batch_linear(x))


def test_convert_to_linear():
    """ BatchLinear can be converted back to a list of Linear modules. Check that the conversion is done correctly. """
    modules = [torch.nn.Linear(3, 4) for _ in range(5)]
    batch_linear = BatchLinear(modules)
    non_batch_linear = NonBatchLinear(batch_linear.to_linear_modules())

    x = torch.randn(5, 3)
    assert torch.allclose(batch_linear(x), non_batch_linear(x))


def test_convert_to_linear_parameters_unchanged():
    """ """
    modules = [torch.nn.Linear(3, 4) for _ in range(5)]
    batch_linear1 = BatchLinear(modules)
    batch_linear2 = BatchLinear(batch_linear1.to_linear_modules())

    for p1,p2 in zip(batch_linear1.parameters(), batch_linear2.parameters()):
        assert p1.shape == p2.shape
        assert (p1 == p2).all()
