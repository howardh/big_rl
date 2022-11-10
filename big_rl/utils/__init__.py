import os

import torch


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
