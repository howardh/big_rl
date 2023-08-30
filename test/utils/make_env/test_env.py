import pytest

import gymnasium

from big_rl.utils.make_env import make_env
from pydantic import ValidationError


##################################################
# Validation
##################################################


def test_empty_config_should_error():
    config = {}
    with pytest.raises(ValueError):
        make_env([config])


def test_invalid_env_type():
    config = {
        'type': 'InvalidEnvType',
        'envs': [{
            'kwargs': {
                'id': 'FrozenLake-v1',
            },
        }],
    }
    with pytest.raises(ValueError):
        make_env([config])


def test_missing_id():
    config = {
        'envs': [{
            'kwargs': {},
        }],
    }
    with pytest.raises(ValueError):
        make_env([config])


def test_extra_params():
    config = {
        'envs': [{
            'kwargs': {
                'id': 'FrozenLake-v1',
            },
        }],
        'extra_params': None,
    }
    with pytest.raises(ValidationError):
        make_env([config])
