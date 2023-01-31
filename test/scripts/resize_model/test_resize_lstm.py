from scripts.resize_model import resize_ModularPolicy5LSTM as resize
from big_rl.model.model import ModularPolicy5LSTM


def test_resize_no_outputs():
    """ Make sure that the resized model state dict can be loaded into a model initialized with the new size. """
    inputs = {
        'reward': {
            'type': 'ScalarInput',
            'config': {
                'value_size': 1,
            },
        },
    }
    model_small = ModularPolicy5LSTM(
        inputs = inputs,
        outputs = {},
        value_size = 5,
        hidden_size = 8,
    )
    state_dict = resize(model_small, hidden_size=9).state_dict()
    model_big = ModularPolicy5LSTM(
        inputs = inputs,
        outputs = {},
        value_size = 5,
        hidden_size = 9,
    )
    model_big.load_state_dict(state_dict)


def test_resize_with_outputs():
    """ Make sure that the resized model state dict can be loaded into a model initialized with the new size. """
    inputs = {
        'reward': {
            'type': 'ScalarInput',
            'config': {
                'value_size': 1,
            },
        },
    }
    outputs = {
        'value': {
            'type': 'LinearOutput',
            'config': {
                'output_size': 1,
            }
        },
    }
    model_small = ModularPolicy5LSTM(
        inputs = inputs,
        outputs = outputs,
        value_size = 5,
        hidden_size = 8,
    )
    state_dict = resize(model_small, hidden_size=9).state_dict()
    model_big = ModularPolicy5LSTM(
        inputs = inputs,
        outputs = outputs,
        value_size = 5,
        hidden_size = 9,
    )
    model_big.load_state_dict(state_dict)
