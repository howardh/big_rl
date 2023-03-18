import pytest

import torch

from big_rl.model.recurrent_attention_16 import BatchRecurrentAttention16Layer_v2, BatchRecurrentAttention16Layer, NonBatchRecurrentAttention16Layer, RecurrentAttention16


# Utils


def assert_same_output(output1, output2):
    assert len(output1) == len(output2)
    for k in ['query', 'key', 'value', 'state']:
        assert k in output1['gates']
        assert k in output2['gates']
        assert torch.allclose(output1['gates'][k], output2['gates'][k], atol=1e-7)
    for k in ['attn_output', 'attn_output_weights', 'key', 'value']:
        assert k in output1
        assert k in output2
        assert torch.allclose(output1[k], output2[k], atol=1e-7)
    for x,y in zip(output1['state'], output2['state']):
        assert torch.allclose(x, y, atol=1e-7)


def dummy_loss(output):
    loss = torch.tensor(0.)
    for k in ['attn_output', 'attn_output_weights', 'key', 'value']:
        loss += output[k].mean()
    for k in ['query', 'key', 'value', 'state']:
        loss += output['gates'][k].mean()
    for x in output['state']:
        loss += x.mean()
    return loss


# Test conversion between batched and non-batched

@pytest.mark.parametrize('batched_cls', [BatchRecurrentAttention16Layer, BatchRecurrentAttention16Layer_v2])
def test_convert_to_batched_same_output(batched_cls):
    # Convert a batched module to a non-batched module.
    # If they're given the same inputs, they should produce the same outputs.
    nonbatched_module = NonBatchRecurrentAttention16Layer(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        num_modules=2,
    )
    batched_module = batched_cls.from_nonbatched(nonbatched_module)

    # Test that the modules produce the same outputs.
    batched_state = batched_module.init_state(5)
    nonbatched_state = nonbatched_module.init_state(5)
    for bs, nbs in zip(batched_state, nonbatched_state):
        assert (bs == nbs).all()
    state = batched_state

    key = torch.randn([2, 5, 8])
    value = torch.randn(2, 5, 8)
    batched_output = batched_module(state, key, value)
    nonbatched_output = nonbatched_module(state, key, value)

    assert_same_output(batched_output, nonbatched_output)


@pytest.mark.parametrize('batched_cls', [BatchRecurrentAttention16Layer, BatchRecurrentAttention16Layer_v2])
def test_convert_to_nonbatched_same_output(batched_cls):
    # Convert a non-batched module to a batched module.
    # If they're given the same inputs, they should produce the same outputs.
    batched_module = batched_cls(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        num_modules=2,
    )
    nonbatched_module = batched_module.to_nonbatched()

    # Test that the modules produce the same outputs.
    batched_state = batched_module.init_state(5)
    nonbatched_state = nonbatched_module.init_state(5)
    for bs, nbs in zip(batched_state, nonbatched_state):
        assert (bs == nbs).all()
    state = batched_state

    key = torch.randn([2, 5, 8])
    value = torch.randn(2, 5, 8)
    batched_output = batched_module(state, key, value)
    nonbatched_output = nonbatched_module(state, key, value)

    assert_same_output(batched_output, nonbatched_output)


@pytest.mark.parametrize('batched_cls', [BatchRecurrentAttention16Layer, BatchRecurrentAttention16Layer_v2])
def test_convert_to_batched_independence(batched_cls):
    # Convert a non-batched module to a batched module.
    # Modifying the weights of one module should not affect the other.
    nonbatched_module = NonBatchRecurrentAttention16Layer(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        num_modules=2,
    )
    batched_module = batched_cls.from_nonbatched(nonbatched_module)

    key = torch.randn([2, 5, 8])
    value = torch.randn(2, 5, 8)

    # Save output before modifying weights.
    state = nonbatched_module.init_state(5)
    output1 = nonbatched_module(state, key, value)

    # Modify weights.
    optim = torch.optim.Adam(batched_module.parameters(), lr=1.0)
    optim.zero_grad()
    loss = dummy_loss(batched_module(state, key, value))
    loss.backward()
    optim.step()

    # Save output after modifying weights.
    state = nonbatched_module.init_state(5)
    output2 = nonbatched_module(state, key, value)

    # Make sure the outputs are unchanged
    assert_same_output(output1, output2)


@pytest.mark.parametrize('batched_cls', [BatchRecurrentAttention16Layer, BatchRecurrentAttention16Layer_v2])
def test_convert_to_nonbatched_independence(batched_cls):
    # Convert a batched module to a non-batched module.
    # Modifying the weights of one module should not affect the other.
    batched_module = batched_cls(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        num_modules=2,
    )
    nonbatched_module = batched_module.to_nonbatched()

    key = torch.randn([2, 5, 8])
    value = torch.randn(2, 5, 8)

    # Save output before modifying weights.
    state = batched_module.init_state(5)
    output1 = batched_module(state, key, value)

    # Modify weights.
    optim = torch.optim.Adam(nonbatched_module.parameters(), lr=1.0)
    optim.zero_grad()
    loss = dummy_loss(nonbatched_module(state, key, value))
    loss.backward()
    optim.step()

    # Save output after modifying weights.
    state = batched_module.init_state(5)
    output2 = batched_module(state, key, value)

    # Make sure the outputs are unchanged
    assert_same_output(output1, output2)


@pytest.mark.parametrize('batched_cls', [BatchRecurrentAttention16Layer, BatchRecurrentAttention16Layer_v2])
def test_same_gradients_after_conversion(batched_cls):
    # Convert a batched module to a non-batched module.
    # Run a step of gradient descent on both modules.
    # The resulting modules should produce the same outputs if given the same inputs.
    batched_module = batched_cls(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        num_modules=2,
    )
    nonbatched_module = batched_module.to_nonbatched()

    key = torch.randn([2, 5, 8])
    value = torch.randn(2, 5, 8)

    # Modify weights.
    for module in [batched_module, nonbatched_module]:
        state = module.init_state(5)
        optim = torch.optim.SGD(module.parameters(), lr=1.0)
        optim.zero_grad()
        loss = dummy_loss(module(state, key, value))
        loss.backward()
        optim.step()

    # Get their outputs
    batched_state = batched_module.init_state(5)
    batched_output = batched_module(batched_state, key, value)

    nonbatched_state = nonbatched_module.init_state(5)
    nonbatched_output = nonbatched_module(nonbatched_state, key, value)

    # Make sure the outputs are unchanged
    assert_same_output(batched_output, nonbatched_output)


# Test merging and removing modules
def test_remove_modules_no_removals():
    # Make sure `remove_modules` can be called without error and the resulting module works
    module = NonBatchRecurrentAttention16Layer(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        num_modules=2,
    )

    key = torch.randn([2, 5, 8])
    value = torch.randn(2, 5, 8)
    state = module.init_state(5)

    output1 = module(state, key, value)

    # Mask is all True, so everything is kept (nothing removed)
    module.remove_modules([True, True])

    output2 = module(state, key, value)

    # Output should be unchanged because nothing was removed
    assert_same_output(output1, output2)


def test_remove_modules_2_to_1():
    # Make sure `remove_modules` can be called without error and the resulting module works
    module = NonBatchRecurrentAttention16Layer(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        num_modules=2,
    )

    key = torch.randn([2, 5, 8])
    value = torch.randn(2, 5, 8)
    state = module.init_state(5)

    module(state, key, value)

    # Mask is all True, so everything is kept (nothing removed)
    module.remove_modules([False, True])

    state = module.init_state(5)
    module(state, key, value)


def test_merge_modules_no_error():
    # Make sure `merge_modules` can be called without error and the resulting module works
    module1 = NonBatchRecurrentAttention16Layer(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        num_modules=2,
    )
    module2 = NonBatchRecurrentAttention16Layer(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        num_modules=3,
    )

    # Merge modules. Make sure it doesn't error.
    module1.merge(module2)

    # Check that resulting model works
    key = torch.randn([4, 5, 8])
    value = torch.randn(4, 5, 8)
    state = module1.init_state(5)
    module1(state, key, value)


def test_merge_and_remove_roundtrip():
    module1 = NonBatchRecurrentAttention16Layer(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        num_modules=2,
    )
    module2 = NonBatchRecurrentAttention16Layer(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        num_modules=3,
    )

    # Get output before merging
    key = torch.randn([4, 5, 8])
    value = torch.randn(4, 5, 8)
    state = module1.init_state(5)

    output1 = module1(state, key, value)

    # Merge modules
    module1.merge(module2)

    # Remove new modules
    module1.remove_modules([True, True, False, False, False])

    # Get output after roundtrip
    output2 = module1(state, key, value)

    # Make sure the outputs are unchanged
    assert_same_output(output1, output2)


def test_merge_and_remove_roundtrip_interleaved():
    module1 = NonBatchRecurrentAttention16Layer(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        num_modules=2,
    )
    module2 = NonBatchRecurrentAttention16Layer(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        num_modules=3,
    )

    # Get output before merging
    key = torch.randn([4, 5, 8])
    value = torch.randn(4, 5, 8)
    state = module1.init_state(5)

    output1 = module1(state, key, value)

    # Merge modules
    module1.merge(module2, positions=[0, 2, 4])

    # Remove new modules
    module1.remove_modules([False, True, False, True, False])

    # Get output after roundtrip
    output2 = module1(state, key, value)

    # Make sure the outputs are unchanged
    assert_same_output(output1, output2)


# misc

@pytest.mark.parametrize('architecture', [[1], [1,1], [1,2]])
def test_state_shape(architecture):
    # Check that the number of tensors in the state matches `state_size`.
    module = RecurrentAttention16(
        input_size=8,
        key_size=8,
        value_size=8,
        num_heads=4,
        ff_size=[3,7],
        architecture=architecture,
    )

    # Check initial hidden state
    state = module.init_state(5)
    assert len(state) == module.state_size

    # Check hidden state after a step
    key = torch.randn([2, 5, 8])
    value = torch.randn(2, 5, 8)
    output = module(state, key, value)
    state = output['state']
    assert len(state) == module.state_size
