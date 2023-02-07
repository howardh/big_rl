import torch

from big_rl.model.model import BatchLinear, NonBatchLinear


def test_same_as_non_batch():
    modules = [torch.nn.Linear(3, 4) for _ in range(5)]
    batch_linear = BatchLinear(modules)
    non_batch_linear = NonBatchLinear(modules)

    x = torch.randn(5, 3)
    assert torch.allclose(batch_linear(x), non_batch_linear(x))


def test_convert_to_linear():
    """ BatchLinear can be converted back to a list of Linear modules. Check that the conversion is done correctly. """
    modules = [torch.nn.Linear(3, 4) for _ in range(5)]
    batch_linear = BatchLinear(modules)
    non_batch_linear = NonBatchLinear(batch_linear.to_linear_modules())

    x = torch.randn(5, 3)
    assert torch.allclose(batch_linear(x), non_batch_linear(x))
