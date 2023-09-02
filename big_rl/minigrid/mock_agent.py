""" An agent designed to test environments. Only works with 4x4 environments (2x2 if we exclude the walls). It has a built-in policy that allows it to thoroughly explore the room and can be set to pick up objects in a particular order. """

import itertools
from typing import Iterable

import gymnasium

from big_rl.minigrid.envs.env_v2 import MultiRoomEnv_v2


def mock_agent(env: gymnasium.Env, target_sequence: Iterable[tuple[str,str]]):
    if not isinstance(env.unwrapped, MultiRoomEnv_v2):
        raise TypeError(f"env must be a MultiRoomEnv_v2, not {type(env)}")
    ## Check that the environment is a 4x4 gridworld
    #if env.width != 4 or env.height != 4:
    #    raise ValueError('The environment must be a 4x4 gridworld')
    
    # There should be no more than 2 objects
    if env.unwrapped.num_objs > 2:
        raise ValueError('Cannot test environments with more than 2 objects')

    # Initialize sequence of actions
    """
    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |
    """
    ACTION_SEQUENCE = [2, 1, 1, 1] # Move forward, turn left 3 times. This should cover the entirety of a 2x2 space with two objects.
    
    # Step through the environment
    done = False
    action_idx = 0

    env.reset()
    done = False
    for target_type, target_color in target_sequence:
        # Check that the target object exists
        if not any([obj.type == target_type and obj.color == target_color for obj in env.unwrapped.objects]):
            raise ValueError(f'The target object ({target_type}, {target_color}) does not exist in the environment')
        for steps_since_last_pickup in itertools.count():
            # Get the position in front of the agent
            fwd_pos = env.unwrapped.front_pos
            # Get the contents of the cell in front of the agent
            fwd_cell = env.unwrapped.grid.get(*fwd_pos)

            # If there's an object in front of the agent and it's the target object, pick it up
            if fwd_cell and fwd_cell.type == target_type and fwd_cell.color == target_color:
                output = env.step(3)
                done = output[2] or output[3]
                yield output
                if done:
                    return
                break
            else:
                output = env.step(ACTION_SEQUENCE[action_idx])
                done = output[2] or output[3]
                yield output
                if done:
                    return
                action_idx = (action_idx + 1) % len(ACTION_SEQUENCE)

            if steps_since_last_pickup > 20:
                raise Exception('The agent is stuck in a loop. Are you sure the environment is a 4x4 grid?')
