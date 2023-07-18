import pytest

import torch

from big_rl.model.core_module.container import CoreModule, CoreModuleParallel, CoreModuleSeries


class DummyCoreModule(CoreModule):
    def __init__(self, key_size: int = 16, value_size: int = 16, num_heads: int = 1, num_outputs: int = 1, hidden_shapes: list[int] = []):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.num_outputs = num_outputs
        self.hidden = torch.nn.ParameterList([torch.randn(1, size) for size in hidden_shapes])

    def forward(self, key, value, hidden):
        batch_size = key.shape[1]

        # Verify that the hidden state is sent to the correct module.
        # We randomly generate a different hidden state for each module and they never change, so we can check this by comparing the values to `self.hidden`
        assert len(hidden) == len(self.hidden)
        for h1, h2 in zip(hidden, self.hidden):
            assert torch.all(torch.eq(h1, h2))

        return {
            'key': torch.ones([self.num_outputs, batch_size, self.key_size]),
            'value': torch.ones([self.num_outputs, batch_size, self.value_size]),
            'hidden': hidden,
        }

    def init_hidden(self, batch_size):
        return tuple([
            h.view(1, -1).expand(batch_size, -1)
            for h in self.hidden
        ])

    @property
    def n_hidden(self):
        return len(self.hidden)


@pytest.mark.parametrize('num_inputs', [1, 2])
@pytest.mark.parametrize('batch_size', [1, 2, 3])
def test_dummy_module(num_inputs, batch_size):
    """ Make sure the dummy module used for testing works as expected. """
    model = DummyCoreModule()

    hidden = model.init_hidden(batch_size)
    key = torch.randn(num_inputs, batch_size, 16)
    value = torch.randn(num_inputs, batch_size, 16)
    for _ in range(10):
        output = model(key, value, hidden)
        hidden = output['hidden']


@pytest.mark.parametrize('num_inputs', [1, 2])
def test_parallel(num_inputs):
    """ Test that parallel core modules work. """
    batch_size = 1
    model = CoreModuleParallel([
        DummyCoreModule(),
        DummyCoreModule(),
    ])
    hidden = model.init_hidden(batch_size)
    key = torch.randn(num_inputs, batch_size, 16)
    value = torch.randn(num_inputs, batch_size, 16)
    for _ in range(10):
        output = model(key, value, hidden)
        hidden = output['hidden']


@pytest.mark.parametrize('num_inputs', [1, 2])
def test_series(num_inputs):
    """ Test that series core modules work. """
    batch_size = 1
    model = CoreModuleSeries([
        DummyCoreModule(),
        DummyCoreModule(),
    ])
    hidden = model.init_hidden(batch_size)
    key = torch.randn(num_inputs, batch_size, 16)
    value = torch.randn(num_inputs, batch_size, 16)
    for _ in range(10):
        output = model(key, value, hidden)
        hidden = output['hidden']


@pytest.mark.parametrize('batch_size', [1, 2, 3])
def test_parallel_hidden_state(batch_size):
    """ Make sure that the hidden states are sent to the correct modules. The assertion is in DummyCoreModule.forward(). """
    num_inputs = 1
    model = CoreModuleParallel([
        DummyCoreModule(hidden_shapes=[3]),
        DummyCoreModule(hidden_shapes=[7, 11]),
        DummyCoreModule(hidden_shapes=[13, 17, 19]),
    ])

    hidden = model.init_hidden(batch_size)

    # Type checking: hidden state should be a tuple of tensors
    assert isinstance(hidden, tuple)
    assert all([isinstance(h, torch.Tensor) for h in hidden])

    # Check length
    assert len(hidden) == model.n_hidden

    key = torch.randn(num_inputs, batch_size, 16)
    value = torch.randn(num_inputs, batch_size, 16)
    for _ in range(10):
        output = model(key, value, hidden)
        hidden = output['hidden']

        # Type checking: hidden state should be a tuple of tensors
        assert isinstance(hidden, tuple)
        assert all([isinstance(h, torch.Tensor) for h in hidden])

        # Check length
        assert len(hidden) == model.n_hidden

    # Make sure that we get an error if the hidden states are sent to the wrong module (just to make sure the test is working)
    hidden = tuple([hidden[1], hidden[0], *hidden[2:]])
    with pytest.raises(Exception):
        output = model(key, value, hidden)


def test_series_hidden_state():
    """ Make sure that the hidden states are sent to the correct modules. The assertion is in DummyCoreModule.forward(). """
    batch_size = 1
    num_inputs = 1
    model = CoreModuleSeries([
        DummyCoreModule(hidden_shapes=[3]),
        DummyCoreModule(hidden_shapes=[7, 11]),
        DummyCoreModule(hidden_shapes=[13, 17, 19]),
    ])

    hidden = model.init_hidden(batch_size)

    # Type checking: hidden state should be a tuple of tensors
    assert isinstance(hidden, tuple)
    assert all([isinstance(h, torch.Tensor) for h in hidden])

    # Check length
    assert len(hidden) == model.n_hidden

    key = torch.randn(num_inputs, batch_size, 16)
    value = torch.randn(num_inputs, batch_size, 16)
    for _ in range(10):
        output = model(key, value, hidden)
        hidden = output['hidden']

        # Type checking: hidden state should be a tuple of tensors
        assert isinstance(hidden, tuple)
        assert all([isinstance(h, torch.Tensor) for h in hidden])

        # Check length
        assert len(hidden) == model.n_hidden

    # Make sure that we get an error if the hidden states are sent to the wrong module (just to make sure the test is working)
    hidden = tuple([hidden[1], hidden[0], *hidden[2:]])
    with pytest.raises(Exception):
        output = model(key, value, hidden)


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('num_modules', [1, 2, 3])
def test_parallel_output_size(batch_size, num_modules):
    num_inputs = 1
    num_modules = 2

    modules: list[CoreModule] = [
        DummyCoreModule(hidden_shapes=[3]),
        DummyCoreModule(hidden_shapes=[7, 11]),
        DummyCoreModule(hidden_shapes=[13, 17, 19]),
    ]

    model = CoreModuleParallel(modules[:num_modules])

    hidden = model.init_hidden(batch_size)
    key = torch.randn(num_inputs, batch_size, 16)
    value = torch.randn(num_inputs, batch_size, 16)
    for _ in range(3):
        output = model(key, value, hidden)
        hidden = output['hidden']

        assert output['key'].shape == (num_modules, batch_size, 16)
        assert output['value'].shape == (num_modules, batch_size, 16)


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('num_modules', [1, 2, 3])
def test_series_output_size(batch_size, num_modules):
    num_inputs = 1

    modules: list[CoreModule] = [
        DummyCoreModule(hidden_shapes=[3]),
        DummyCoreModule(hidden_shapes=[7, 11]),
        DummyCoreModule(hidden_shapes=[13, 17, 19]),
    ]

    model = CoreModuleSeries(modules[:num_modules])

    hidden = model.init_hidden(batch_size)
    key = torch.randn(num_inputs, batch_size, 16)
    value = torch.randn(num_inputs, batch_size, 16)
    for _ in range(3):
        output = model(key, value, hidden)
        hidden = output['hidden']

        assert output['key'].shape == (1, batch_size, 16)
        assert output['value'].shape == (1, batch_size, 16)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_nested_modules(batch_size):
    """ If we make a model consisting of nested CoreModuleParallel and CoreModuleSeries and plain core modules, it sould run without error. """
    num_inputs = 10

    modules: list[CoreModule] = [
        DummyCoreModule(hidden_shapes=[3]),
        DummyCoreModule(hidden_shapes=[7, 11]),
        DummyCoreModule(hidden_shapes=[13, 17, 19]),
    ]

    model = CoreModuleParallel([
        CoreModuleSeries(modules[:1]),
        CoreModuleSeries(modules),
        DummyCoreModule(hidden_shapes=[3]),
    ])

    hidden = model.init_hidden(batch_size)
    key = torch.randn(num_inputs, batch_size, 16)
    value = torch.randn(num_inputs, batch_size, 16)
    for _ in range(3):
        output = model(key, value, hidden)
        hidden = output['hidden']

        # 3 parallel modules, each produce 1 output, so there should be 3 outputs total (dim 1)
        assert output['key'].shape == (3, batch_size, 16)
        assert output['value'].shape == (3, batch_size, 16)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_nested_modules_multiple_outputs(batch_size):
    """ If we make a model consisting of nested CoreModuleParallel and CoreModuleSeries and plain core modules, it sould run without error.
    """
    num_inputs = 10

    modules: list[CoreModule] = [
        DummyCoreModule(hidden_shapes=[3], num_outputs=1),
        DummyCoreModule(hidden_shapes=[7, 11], num_outputs=3),
        DummyCoreModule(hidden_shapes=[13, 17, 19], num_outputs=2),
    ]

    model = CoreModuleParallel([
        CoreModuleSeries(modules[:1]),  # 1 output
        CoreModuleSeries(modules),  # 2 outputs
        DummyCoreModule(hidden_shapes=[3]),  # 1 output
    ])

    hidden = model.init_hidden(batch_size)
    key = torch.randn(num_inputs, batch_size, 16)
    value = torch.randn(num_inputs, batch_size, 16)
    for _ in range(3):
        output = model(key, value, hidden)
        hidden = output['hidden']

        assert output['key'].shape == (4, batch_size, 16)
        assert output['value'].shape == (4, batch_size, 16)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_deeper_nested_modules(batch_size):
    """ If we make a model consisting of nested CoreModuleParallel and CoreModuleSeries and plain core modules, it sould run without error.
    """
    num_inputs = 10

    modules: list[CoreModule] = [
        DummyCoreModule(hidden_shapes=[3]),
        DummyCoreModule(hidden_shapes=[7, 11]),
        DummyCoreModule(hidden_shapes=[13, 17, 19]),
    ]

    model = CoreModuleParallel([
        CoreModuleSeries([
            CoreModuleParallel([
                CoreModuleSeries(modules[:1]),
                DummyCoreModule(hidden_shapes=[3]),
                CoreModuleSeries(modules),
            ]),
            DummyCoreModule(hidden_shapes=[3]),
        ]),  # 1 output
        CoreModuleSeries([
            CoreModuleSeries(modules[:1]),
            CoreModuleSeries(modules[:1]),
        ]),  # 1 output
        DummyCoreModule(hidden_shapes=[3]),  # 1 output
    ])

    hidden = model.init_hidden(batch_size)
    key = torch.randn(num_inputs, batch_size, 16)
    value = torch.randn(num_inputs, batch_size, 16)
    for _ in range(3):
        output = model(key, value, hidden)
        hidden = output['hidden']

        assert output['key'].shape == (3, batch_size, 16)
        assert output['value'].shape == (3, batch_size, 16)


def test_module_has_parameters():
    """ Regression test: Sub-modules were kept as a plain list rather than a ModuleList, so gradients were not passed to them and they were not moved to the GPU. """
    model = CoreModuleParallel([
        CoreModuleSeries([
            DummyCoreModule(hidden_shapes=[3]),
            DummyCoreModule(hidden_shapes=[4]),
        ]),
        CoreModuleParallel([
            DummyCoreModule(hidden_shapes=[5]),
            DummyCoreModule(hidden_shapes=[6]),
        ]),
        DummyCoreModule(hidden_shapes=[7]),
    ])

    assert len(model.state_dict()) > 0
