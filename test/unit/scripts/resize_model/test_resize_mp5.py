import pytest

from scripts.resize_model import resize_ModularPolicy5 as resize
from big_rl.model.model import ModularPolicy5


# @pytest.mark.parametrize('recurrence_type', ['RecurrentAttention11', 'RecurrentAttention14'])
@pytest.mark.parametrize('recurrence_type', ['RecurrentAttention14'])
def test_resize_1_layer_grow(recurrence_type):
    """ Make sure that the resized model state dict can be loaded into a model initialized with the new size. """
    inputs = {
        'reward': {
            'type': 'ScalarInput',
            'config': {},
        },
    }
    params = {
        'inputs': inputs,
        'outputs': {},
        'num_heads': 1,
        'input_size': 2,
        'key_size': 3,
        'value_size': 5,
        'ff_size': 7,
        'recurrence_type': recurrence_type,
    }
    model_small = ModularPolicy5(
        **params,
        architecture=[3],
    )
    state_dict = resize(model_small, architecture=[4]).state_dict()
    model_big = ModularPolicy5(
        **params,
        architecture=[4]
    )
    model_big.load_state_dict(state_dict)


def test_resize_2_layers_grow():
    """ Make sure that the resized model state dict can be loaded into a model initialized with the new size. """
    inputs = {
        'reward': {
            'type': 'ScalarInput',
            'config': {},
        },
    }
    params = {
        'inputs': inputs,
        'outputs': {},
        'num_heads': 1,
        'input_size': 2,
        'key_size': 3,
        'value_size': 5,
        'ff_size': 7,
        'recurrence_type': 'RecurrentAttention14',
    }
    model_small = ModularPolicy5(
        **params,
        architecture=[3, 4],
    )
    state_dict = resize(model_small, architecture=[5, 6]).state_dict()
    model_big = ModularPolicy5(
        **params,
        architecture=[5, 6]
    )
    model_big.load_state_dict(state_dict)


def test_grow_and_shrink_unchanged():
    """ Resize a model twice, growing it the first time and shrinking it the second time back to the original size. The model should be unchanged. """
    inputs = {
        'reward': {
            'type': 'ScalarInput',
            'config': {},
        },
    }
    params = {
        'inputs': inputs,
        'outputs': {},
        'num_heads': 1,
        'input_size': 2,
        'key_size': 3,
        'value_size': 5,
        'ff_size': 7,
        'recurrence_type': 'RecurrentAttention14',
    }
    model0 = ModularPolicy5(
        **params,
        architecture=[3],
    )
    state_dict0 = resize(model0, architecture=[4]).state_dict()

    model1 = ModularPolicy5(
        **params,
        architecture=[4]
    )
    model1.load_state_dict(state_dict0)
    state_dict1 = resize(model1, architecture=[3]).state_dict()

    model2 = ModularPolicy5(
        **params,
        architecture=[3],
    )
    model2.load_state_dict(state_dict1)

    # Check that model0 and model2 are the same
    for p0, p2 in zip(model0.parameters(), model2.parameters()):
        assert p0.shape == p2.shape
        assert (p0 == p2).all()
