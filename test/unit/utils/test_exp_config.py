import pytest

from big_rl.utils import ExperimentConfigs, ConfigReplace, ConfigDelete


def test_add():
    config = ExperimentConfigs()
    config.add('config-1', {'a': 1})
    config.add('config-2', {'b': 1})

    assert 'config-1' in config.keys()
    assert 'config-2' in config.keys()

    assert config['config-1'] == {'a': 1}
    assert config['config-2'] == {'b': 1}


def test_add_change():
    config = ExperimentConfigs()
    config.add('config-1', {'a': 1})
    config.add_change('config-2', {'b': 1})

    assert 'config-1' in config.keys()
    assert 'config-2' in config.keys()

    assert config['config-1'] == {'a': 1}
    assert config['config-2'] == {'a': 1, 'b': 1}


def test_repeat_config_name_errors():
    config = ExperimentConfigs()
    config.add('config-1', {'a': 1})

    with pytest.raises(Exception):
        config.add('config-1', {'b': 2})


def test_add_change_nested_dict_replace_one_entry():
    config = ExperimentConfigs()
    config.add('config-1', {'a': {'b': 2, 'c': 3}, 'd': 4})
    config.add_change('config-2', {'a': {'b': 5}})

    assert 'config-1' in config.keys()
    assert 'config-2' in config.keys()

    assert config['config-1'] == {'a': {'b': 2, 'c': 3}, 'd': 4}
    assert config['config-2'] == {'a': {'b': 5, 'c': 3}, 'd': 4}


def test_add_change_nested_dict_add_entry():
    config = ExperimentConfigs()
    config.add('config-1', {'a': {'b': 2, 'c': 3}})
    config.add_change('config-2', {'a': {'d': 4}})

    assert 'config-1' in config.keys()
    assert 'config-2' in config.keys()

    assert config['config-1'] == {'a': {'b': 2, 'c': 3}}
    assert config['config-2'] == {'a': {'b': 2, 'c': 3, 'd': 4}}


def test_add_change_lists():
    """ Lists of ints are completely replaced by default """
    config = ExperimentConfigs()
    config.add('config-1', {'a': [1, 2, 3]})
    config.add_change('config-2', {'a': [4, 5, 6]})

    assert 'config-1' in config.keys()
    assert 'config-2' in config.keys()

    assert config['config-1'] == {'a': [1, 2, 3]}
    assert config['config-2'] == {'a': [4, 5, 6]}


def test_add_change_merge_list_of_dicts():
    """ Lists of dicts are merged by default """
    config = ExperimentConfigs()
    config.add('config-1', {'a': [{'b': 2, 'c': 3}, {'d': 4}]})
    config.add_change('config-2', {'a': [{'e': 5}, {'f': 6}]})

    assert 'config-1' in config.keys()
    assert 'config-2' in config.keys()

    assert config['config-1'] == {'a': [{'b': 2, 'c': 3}, {'d': 4}]}
    assert config['config-2'] == {'a': [{'b': 2, 'c': 3, 'e': 5}, {'d': 4, 'f': 6}]}


def test_add_change_merge_list_of_dicts_and_ints():
    """ Heterogeneous lists are merged if they are the same length. If the entries are different types, they are replaced with the new value. """
    config = ExperimentConfigs()
    config.add('config-1', {'a': [0, {'b': 2}]})
    config.add_change('config-2', {'a': [1, {'c': 3}]})
    config.add_change('config-3', {'a': [{'d': 4}, {'b': 0}]})
    config.add_change('config-4', {'a': [{}, 1]})

    assert 'config-1' in config.keys()
    assert 'config-2' in config.keys()

    assert config['config-1'] == {'a': [0, {'b': 2}]}
    assert config['config-2'] == {'a': [1, {'b': 2, 'c': 3}]}
    assert config['config-3'] == {'a': [{'d': 4}, {'b': 0, 'c': 3}]}
    assert config['config-4'] == {'a': [{'d': 4}, 1]}


def test_add_change_replace_list_of_dicts_of_different_length():
    """ Lists of dicts of different lengths are not merged """
    config = ExperimentConfigs()
    config.add('config-1', {'a': [{'b': 2, 'c': 3}, {'d': 4}, {'e': 5}]})
    config.add_change('config-2', {'a': [{'e': 5}, {'f': 6}]})

    assert 'config-1' in config.keys()
    assert 'config-2' in config.keys()

    assert config['config-1'] == {'a': [{'b': 2, 'c': 3}, {'d': 4}, {'e': 5}]}
    assert config['config-2'] == {'a': [{'e': 5}, {'f': 6}]}


def test_add_change_new_lists():
    """ New lists are added to the dict. """
    config = ExperimentConfigs()
    config.add('config-1', {})
    config.add_change('config-2', {'a': [4, 5, 6]})

    assert 'config-1' in config.keys()
    assert 'config-2' in config.keys()

    assert config['config-1'] == {}
    assert config['config-2'] == {'a': [4, 5, 6]}


# Overriding default behaviour


def test_add_change_replace_dict():
    """ Lists are completely replaced by default """
    config = ExperimentConfigs()
    config.add('config-1', {'a': {'b': 2, 'c': 3}, 'd': 4})
    config.add_change('config-2', {'a': ConfigReplace({'e': 5})})

    assert 'config-1' in config.keys()
    assert 'config-2' in config.keys()

    assert config['config-1'] == {'a': {'b': 2, 'c': 3}, 'd': 4}
    assert config['config-2'] == {'a': {'e': 5}, 'd': 4}


def test_add_change_delete_key():
    """ Lists are completely replaced by default """
    config = ExperimentConfigs()
    config.add('config-1', {'a': {'b': 2, 'c': 3}, 'd': 4})
    config.add_change('config-2', {'a': {'b': ConfigDelete()}})
    config.add_change('config-3', {'a': ConfigDelete(), 'd': ConfigDelete()})

    assert 'config-1' in config.keys()
    assert 'config-2' in config.keys()

    assert config['config-1'] == {'a': {'b': 2, 'c': 3}, 'd': 4}
    assert config['config-2'] == {'a': {'c': 3}, 'd': 4}
    assert config['config-3'] == {}
