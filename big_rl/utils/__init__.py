import copy
import os
import time
from typing import Sequence, Iterable, Union, Tuple, Mapping

import numpy as np
import gymnasium
import gymnasium.spaces
import torch
import scipy.interpolate


def torch_save(obj, path):
    """ Save a torch object to a file. Save as `torch.save(obj, path)`, but if the file already exists, the data is written to a separate file, then renamed to take the place of the old file. This is to avoid problems written file. """
    torch.save(obj, path + '.tmp')
    os.rename(path + '.tmp', path)


def validate_checkpoint(obj):
    """ Check if a checkpoint is valid. """
    if not isinstance(obj, dict):
        return False
    for key in obj.keys():
        if key not in ['model', 'optimizer']:
            return False
    return True


def merge_space(*spaces):
    if isinstance(spaces[0], gymnasium.spaces.Dict):
        new_space = {}
        for space in spaces:
            for k,v in space.items():
                if k in new_space:
                    if new_space[k] != v:
                        # Found keys with different spaces
                        # If they are the same shape, then we can merge them
                        if isinstance(new_space[k], gymnasium.spaces.Box) and isinstance(v, gymnasium.spaces.Box) and new_space[k].shape == v.shape:
                            new_space[k] = gymnasium.spaces.Box(low=v.low * np.nan, high=v.high * np.nan)
                        else:
                            #raise ValueError(f"Space mismatch for key {k}: {new_space[k]} != {v}")
                            return None
                else:
                    new_space[k] = v
        return gymnasium.spaces.Dict(new_space)
    elif isinstance(spaces[0], gymnasium.spaces.Discrete):
        n = spaces[0].n
        for space in spaces:
            if space.n != n:
                #raise ValueError(f"Space mismatch: {space.n} != {n}")
                return None
        return gymnasium.spaces.Discrete(n)
    elif isinstance(spaces[0], gymnasium.spaces.Box):
        low = spaces[0].low
        high = spaces[0].high
        mismatched_low = False
        mismatched_high = False
        for space in spaces:
            if (space.low != low).any():
                mismatched_low = True
            if (space.high != high).any():
                mismatched_high = True
        return gymnasium.spaces.Box(
            low * np.nan if mismatched_low else low,
            high * np.nan if mismatched_high else high,
        )
    else:
        raise NotImplementedError(f"Space type {type(spaces[0])} not supported")


def zip2(*args) -> Iterable[Union[Tuple,Mapping]]:
    """
    Zip objects together. If dictionaries are provided, the lists within the dictionary are zipped together.

    >>> list(zip2([1,2,3], [4,5,6]))
    [(1, 4), (2, 5), (3, 6)]

    >>> list(zip2({'a': [4,5,6], 'b': [7,8,9]}))
    [{'a': 4, 'b': 7}, {'a': 5, 'b': 8}, {'a': 6, 'b': 9}]

    >>> list(zip2([1,2,3], {'a': [4,5,6], 'b': [7,8,9]}))
    [(1, {'a': 4, 'b': 7}), (2, {'a': 5, 'b': 8}), (3, {'a': 6, 'b': 9})]

    >>> import torch
    >>> list(zip2(torch.tensor([1,2,3]), torch.tensor([4,5,6])))
    [(tensor(1), tensor(4)), (tensor(2), tensor(5)), (tensor(3), tensor(6))]
    """
    if len(args) == 1:
        if isinstance(args[0],(Sequence)):
            return args[0]
        if isinstance(args[0],torch.Tensor):
            return (x for x in args[0])
        if isinstance(args[0], dict):
            keys = args[0].keys()
            return (dict(zip(keys, vals)) for vals in zip(*(args[0][k] for k in keys)))
    return zip(*[zip2(a) for a in args])


##################################################
# File IO
##################################################


def get_results_root_directory(temp=False):
    """ Get the directory where results are to be stored. """
    host_name = os.uname()[1]
    # Mila (Check available resources with `sinfo`)
    mila_hostnames = ['rtx', 'leto', 'eos', 'bart', 'mila', 'kepler', 'power', 'apollor', 'apollov', 'cn-', 'login-1', 'login-2', 'login-3', 'login-4']
    if host_name.endswith('server.mila.quebec') or any((host_name.startswith(n) for n in mila_hostnames)):
        if temp:
            return "/miniscratch/huanghow"
        else:
            return "/network/projects/h/huanghow"
    # RL Lab
    if host_name == "agent-server-1" or host_name == "agent-server-2":
        return "/home/ml/users/hhuang63/results"
        #return "/NOBACKUP/hhuang63/results"
    if host_name == "garden-path" or host_name == "ppl-3":
        return "/home/ml/hhuang63/results"
    # Compute Canada
    if host_name.find('gra') == 0 or host_name.find('cdr') == 0:
        return "/home/hhuang63/scratch/results"
    # Local
    if host_name.find('howard-pc') == 0:
        return "/home/howard/tmp/results"
    # Travis
    if host_name.startswith('travis-'):
        return './tmp'
    raise NotImplementedError("No default path defined for %s" % host_name)


def is_slurm():
    return 'SLURM_JOB_ID' in os.environ


def generate_id(slurm_split: bool = False) -> str:
    """ Generate an identifier that is unique to the current run. If the script is run as a Slurm job, the job ID is used. Otherwise, the current date and time are used.

    Args:
        slurm_split (bool): Set to True if the Slurm job is a single job split into multiple parts as an array job. This will give the same identifier to all jobs of the array.
    """
    slurm_job_id = os.environ.get('SLURM_JOB_ID') # Unique per job
    slurm_array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID') # Same for every job in an array
    slurm_array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID') # Unique per job in the array

    if slurm_job_id is not None: # If it's a slurm job, use the job ID in the directory name
        if slurm_array_job_id is not None:
            # `slurm_array_job_id` is None if it is not an array job
            if slurm_split:
                # If `slurm_split` is True, that means we want to run one experiment split over multiple jobs in an array, so every job in the array should have the same `trial_id`.
                return f'{slurm_array_job_id}'
            else:
                return f'{slurm_array_job_id}_{slurm_array_task_id}'
        else:
            return f'{slurm_job_id}'
    else: # If it is not a slurm job, use the data/time to name the directory
        return time.strftime("%Y_%m_%d-%H_%M_%S")


def create_unique_file(directory, name, extension):
    """ Create a unique file in the given directory with the given name and extension. If the file already exists, a number is appended to the name.

    Returns the full path to the file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    if extension.startswith('.'):
        extension = extension[1:]
    filename = os.path.join(directory, f'{name}.{extension}')
    index = 0
    while True:
        try: 
            # Create the file if it doesn't exist
            # This is atomic on POSIX systems (https://linux.die.net/man/3/open says "The check for the existence of the file and the creation of the file if it does not exist shall be atomic with respect to other threads executing open() naming the same filename in the same directory with O_EXCL and O_CREAT set")
            # XXX: Not sure if this is atomic on a non-POSIX filesystem
            f = os.open(filename,  os.O_CREAT | os.O_EXCL)
            os.close(f)
        except FileExistsError:
            index += 1
            filename = os.path.join(directory, f'{name}-{index}.{extension}')
            continue
        return filename


def create_unique_directory(directory, name):
    """ Create a unique subdirectory in the given directory with the given name. If the directory already exists, a number is appended to the name.

    Returns the full path to the new subdirectory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, f'{name}')
    index = 0
    while True:
        try: 
            # Create the directory if it doesn't exist
            # XXX: Not sure about atomicity of this operation
            os.mkdir(filename)
        except FileExistsError:
            index += 1
            filename = os.path.join(directory, f'{name}-{index}')
            continue
        return filename


##################################################
# Experiment Configs
##################################################


def merge(source, destination):
    """
    (Source: https://stackoverflow.com/a/20666342/382388)

    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    destination = copy.deepcopy(destination)
    if not isinstance(source, type(destination)):
        return source
    if isinstance(source, Mapping):
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                destination[key] = merge(value, node)
            elif isinstance(value, list):
                if key in destination and isinstance(destination[key],list) and len(destination[key]) == len(value):
                    destination[key] = [merge(s,d) for s,d in zip(source[key],destination[key])]
                else:
                    destination[key] = value
            elif isinstance(value, ConfigReplace):
                destination[key] = value.value
            elif isinstance(value, ConfigDelete):
                del destination[key]
            else:
                destination[key] = value

        return destination
    else:
        return source


class ExperimentConfigs(dict):
    """ A dictionary of experiment configurations. The main purpose of this class is to allow configs to be written in a way where it is easier to see the differences from one set of configs to another, and to avoid bugs arising from copy-pasting configurations and forgetting to change their names.

    New configurations can be added as complete dictionaries, or as partial dictionaries that are merged with the existing configurations. By default, dicts and lists of dicts (if both are the same length) are merged recursively. All other types are completely replaced with the new value. This behaviour can be overridden by wrapping the object with `ConfigMerge` or `ConfigReplace`. Keys can also be deleted with `ConfigDelete`.
    """
    def __init__(self):
        self._last_key = None
    def add(self, key, config, inherit=None):
        if key in self:
            raise Exception(f'Key {key} already exists.')
        if inherit is None:
            self[key] = config
        else:
            self[key] = merge(config,self[inherit])
        self._last_key = key
    def add_change(self, key, config):
        self.add(key, config, inherit=self._last_key)


class ConfigReplace:
    def __init__(self, value):
        self.value = value


class ConfigMerge:
    def __init__(self, value):
        if not isinstance(value,dict):
            raise Exception('ConfigMerge can only be used with dicts')
        self.value = value


class ConfigDelete:
    ...


##################################################
# Data Processing
##################################################

def resample_series(data: list[tuple[list,list]], truncate: bool = False) -> tuple[list[float], list[list[float]]]:
    """ Given a list of time-series data with different indepdent variables (i.e. x), interpolate the data to have the same independent variables.

    Args:
        data (list[tuple[list[float],list[float]]): A list of tuples where the first element is the x values and the second element is the y values.
        truncate (bool): If True, truncate the longer curves to the length of the shortest curve. If False, pad the shorter curves with NaNs.
    """

    # Validation
    # Each x-y pair must have matching lengths
    for i,(x,y) in enumerate(data):
        if len(x) != len(y):
            raise ValueError(f"Data pair {i} has mismatched lengths: {len(x) = }, {len(y) = }")

    # Check if all curves have the same length
    # If not, then truncate the longer curves
    if truncate:
        max_x = min(max(curve[0]) for curve in data)
        selected_indices = [np.array(curve[0]) <= max_x for curve in data]
        x_truncated = [np.array(curve[0])[indices] for curve,indices in zip(data,selected_indices)]
        y_truncated = [np.array(curve[1])[indices] for curve,indices in zip(data,selected_indices)]
    else:
        x_truncated = [curve[0] for curve in data]
        y_truncated = [curve[1] for curve in data]

    # Collect the unique x values
    x_set = set()
    for x_ in x_truncated:
        x_set.update(x_)
    x_interp = sorted(list(x_set))
    y_interp = []
    for x,y in zip(x_truncated,y_truncated):
        lin = scipy.interpolate.interp1d(x, y, kind='linear')
        y_interp.append(lin(x_interp))
    return x_interp, y_interp
