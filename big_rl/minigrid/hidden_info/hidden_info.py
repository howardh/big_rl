import itertools

import numpy as np
import numpy.typing as npt

#from big_rl.minigrid.envs.env_v2 import MultiRoomEnv_v2

from minigrid.core.constants import IDX_TO_OBJECT, COLOR_NAMES
from minigrid.wrappers import OBJECT_TO_IDX

"""
See https://github.com/Farama-Foundation/Minigrid/blob/5d49aa0e1aef7a6b7eeac549bb5cebadcb94b99e/minigrid/minigrid_env.py#L175

    AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

    OBJECT_TO_STR = {
        "wall": "W",
        "floor": "F",
        "door": "D",
        "key": "K",
        "ball": "A",
        "box": "B",
        "goal": "G",
        "lava": "V",
    }

"""

OBJ_TYPES = ['key', 'ball']
OBJ_COLORS = COLOR_NAMES
OBJECTS = list(itertools.product(OBJ_COLORS, OBJ_TYPES))
OBJ_TO_IDX = {obj: i for i, obj in enumerate(OBJECTS)}


def get_target_objs(env):
    """ Return a list of target objects """
    if type(env.unwrapped).__name__ == 'MultiRoomEnv_v1':
        for obj in env.objects:
            if obj.type == env.target_type and obj.color == env.target_color:
                return [obj]
        raise ValueError("No target object found")
    if type(env.unwrapped).__name__ == 'MultiRoomEnv_v2':
        return env.task.target_objs
    raise NotImplementedError(f'Unkown environment type: {type(env).__name__}')


def get_target_obj_idx(env):
    """ Return the index associated with this objects type and color. Only works if there is exactly one target. """
    if type(env.unwrapped).__name__ == 'MultiRoomEnv_v1':
        return OBJ_TO_IDX[(env.target_color, env.target_type)]
    if type(env.unwrapped).__name__ == 'MultiRoomEnv_v2':
        assert len(env.task.target_objs) == 1
        return OBJ_TO_IDX[(env.task.target_objs[0].color, env.task.target_objs[0].type)]
    raise NotImplementedError(f'Unkown environment type: {type(env).__name__}')


def get_target_obj_relative_pos(env):
    """ Return the target object's position relative to the agent. Only works if there is exactly one target. """
    if type(env.unwrapped).__name__ == 'MultiRoomEnv_v1':
        return OBJ_TO_IDX[(env.target_color, env.target_type)]
    if type(env.unwrapped).__name__ == 'MultiRoomEnv_v2':
        assert len(env.task.target_objs) == 1
        return OBJ_TO_IDX[(env.task.target_objs[0].color, env.task.target_objs[0].type)]
    raise NotImplementedError(f'Unkown environment type: {type(env).__name__}')


def get_target_obj_vector(env) -> npt.NDArray:
    """ Return a vector of the target object """
    ...


def get_relative_target_obj_pos(env) -> npt.NDArray:
    """ Return a vector of the target object """
    agent_pos = env.agent_pos
    front_vec = env.front_pos - agent_pos
    right_vec = env.right_vec
    target_objs = get_target_objs(env)
    assert len(target_objs) == 1
    obj_pos = target_objs[0].cur_pos

    """
    x*right_vec + y*front_vec = obj_pos - agent_pos
    x*right_vec[0] + y*front_vec[0] = obj_pos[0] - agent_pos[0]
    x*right_vec[1] + y*front_vec[1] = obj_pos[1] - agent_pos[1]
    """

    A = np.array([[right_vec[0], front_vec[0]], [right_vec[1], front_vec[1]]])
    b = np.array([obj_pos[0] - agent_pos[0], obj_pos[1] - agent_pos[1]])
    rel_pos = np.linalg.solve(A, b)

    return rel_pos


def get_all_objects_vector(env) -> npt.NDArray:
    """ Return a one-hot vector of all objects in the environment """
    output = np.zeros(len(OBJECTS), dtype=np.int8)
    for obj in env.objects:
        output[OBJ_TO_IDX[(obj.color, obj.type)]] = 1
    return output


def get_all_objects_pos(env) -> npt.NDArray:
    """ Return a matrix with the position of all objects in the environment. Objects that are not present will have a position of (-1,-1) """
    output = np.ones([len(OBJECTS),2], dtype=np.int8) * -1
    for obj in env.objects:
        output[OBJ_TO_IDX[(obj.color, obj.type)],:] = obj.cur_pos
    return output


def get_relative_wall_map(env, size):
    """ Return a `size` x `size` map of the walls relative to the agent in the form of a binary matrix, where 1 is a wall, and 0 is not a wall. """
    agent_pos = env.agent_pos
    front_vec = env.front_pos - agent_pos
    right_vec = env.right_vec
    wall_map = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            pos = agent_pos + (i - size//2) * right_vec + (j - size//2) * front_vec
            try:
                if env.grid.get(*pos) is not None and env.grid.get(*pos).type == 'wall':
                    wall_map[i, j] = 1
            except:
                pass # Out of bounds, which means the value remains unchanged
    return wall_map
