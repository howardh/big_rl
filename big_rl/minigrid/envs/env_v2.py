from abc import ABC, abstractmethod
from typing import Sequence, TypedDict, Literal, Tuple

import numpy as np
import gymnasium
import gymnasium.spaces
import gymnasium.utils.seeding
import minigrid.wrappers
import minigrid.core
import minigrid.core.mission
import minigrid.core.grid
import minigrid.core.constants
import minigrid.core.world_object
from minigrid.minigrid_env import MiniGridEnv, MissionSpace

from big_rl.minigrid.envs import init_rng, Room, room_is_valid
from big_rl.minigrid.envs import PotentialBasedReward, RewardNoise, RewardDelay, RewardDelayedStart


##################################################
# Tasks
##################################################


class TaskStepReturn(TypedDict):
    reward: float
    trial_completed: bool

    pseudo_reward: float | None
    potential: float | None


class Task(ABC):
    def __init__(self, env: 'MultiRoomEnv_v2'):
        self.env = env

    @abstractmethod
    def reset(self) -> TaskStepReturn: ...

    @abstractmethod
    def step(self) -> TaskStepReturn: ...

    @property
    @abstractmethod
    def description(self) -> str: ...


class NoTask(Task):
    def reset(self):
        return TaskStepReturn(
            reward = 0,
            trial_completed = False,
            pseudo_reward = None,
            potential = None,
        )
    def step(self):
        return TaskStepReturn(
            reward = 0,
            trial_completed = False,
            pseudo_reward = None,
            potential = None,
        )
    @property
    def description(self):
        return 'No task'


class FetchTask(Task):
    def __init__(self,
                 env: 'MultiRoomEnv_v2',
                 rng: np.random.RandomState,
                 num_targets: int = 1,
                 # Reward
                 reward_correct: float = 1,
                 reward_incorrect: float = -1,
                 reward_flip_prob: float = 0.0,
                 reward_type: Literal['standard', 'pbrs'] = 'standard',
                 # Pseudo-reward
                 pseudo_reward_config: dict | None = None,
                 # Potential-based reward shaping
                 pbrs_discount: float = 0.99,
                 pbrs_scale: float = 1.0,
                 # Misc
                 fixed_target: Tuple[str,str] | None = None,
                 cycle_targets: bool = False,
                 ):
        """
        Args:
            env: Environment
            rng: A `np.random.RandomState` or a seed for a RandomState.
            num_targets: The number of objects that yield `reward_correct` rewards.
            reward_correct: Reward to provide if the agent picks up the correct object.
            reward_incorrect: Reward to provide if the agent picks up the incorrect object.
            reward_flip_prob: Probability of giving a reward at all upon completing a trial.
        """
        self.env = env
        self.rng = rng

        self.num_targets = num_targets
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.reward_flip_prob = reward_flip_prob
        self.reward_type = reward_type

        self.pseudo_reward_config = pseudo_reward_config

        self.pbrs = PotentialBasedReward(
            discount=pbrs_discount,
            scale=pbrs_scale,
        )

        self.fixed_target = fixed_target
        self.cycle_targets = cycle_targets

    def reset(self):
        self.objects = self.env.objects

        # Pick a random object
        if self.fixed_target is None:
            self.target_objs = self.rng.choice(self.env.objects, replace=False, size=self.num_targets)
        else:
            self.target_objs = [obj for obj in self.env.objects if obj.color == self.fixed_target[1] and obj.type == self.fixed_target[0]]

        # Choose an order to cycle through the targets
        if self.cycle_targets:
            self.cycle_targets_order = self.rng.permutation(self.target_objs)
            self.cycle_targets_idx = 0

        # Reset statistics
        self.rewards = []
        self.regrets = []

        self.pbrs.reset()

        # Store agent position
        self._agent_pos_prev = self.env.agent_pos

    def step(self) -> TaskStepReturn:
        reward = 0
        trial_completed = False

        # Check if carrying anything
        carrying = self.env.carrying
        if carrying:
            trial_completed = True
            reward_correct = self.reward_correct
            reward_incorrect = self.reward_incorrect
            # Flip the reward with some probability
            p = self.reward_flip_prob
            if self.rng.uniform() < p:
                reward_correct = self.reward_incorrect
                reward_incorrect = self.reward_correct
            # Check if the agent picked up the correct object
            if self._is_target_obj(carrying):
                reward = reward_correct
                regret = 0
                # Cycle to the next target if applicable
                if self.cycle_targets:
                    self.cycle_targets_idx = (self.cycle_targets_idx + 1) % len(self.target_objs)
                    self.target_objs = [self.cycle_targets_order[self.cycle_targets_idx]]
            else:
                reward = reward_incorrect
                regret = reward_correct*(1-p)+reward_incorrect*p - reward_incorrect*(1-p)+reward_correct*p
            self.regrets.append(regret)
            self.rewards.append(reward)

        # PBRS
        pbrs_reward = 0
        if self.reward_type == 'pbrs':
            self.pbrs.update_potential(self._potential())
            pbrs_reward = self.pbrs.reward

        # Save current state for next step
        self._agent_pos_prev = self.env.agent_pos

        return TaskStepReturn(
            reward = reward+pbrs_reward,
            trial_completed = trial_completed,
            pseudo_reward = self._pseudo_reward(),
            potential = self._potential(),
        )

    def _is_target_obj(self, obj):
        for target in self.target_objs:
            if obj.color != target.color:
                continue
            if obj.type != target.type:
                continue
            return True
        return False

    def _get_dests(self):
        return self.target_objs

    def _get_closest_dest(self):
        """ Return the closest destination. If there are no destinations, then return None. """
        destinations = self._get_dests()
        agent_pos = self.env.agent_pos

        closest_distance = float('inf')
        closest_destination = None
        for dest in destinations:
            distance = np.abs(agent_pos-dest.cur_pos).sum()
            if distance > closest_distance:
                continue
            closest_distance = distance
            closest_destination = dest
        return closest_destination

    def _pseudo_reward(self) -> float | None:
        """ Return the pseudo reward for the current state.

        Args:
            config: dictionary containing the following keys:
                - 'reward_type': 'zero', 'inverse distance', 'adjacent to subtask'
                - 'noise': Tuple of (noise type, *noise params). Possible noise types with their parameters:
                    - ('zero', p): Shaped reward is 0 with probability p
                    - ('gaussian', std): Zero-mean Gaussian noise with standard deviation std is added to the shaped reward
                    - ('stop', n): If n>1, the shaped reward is set to 0 after n steps. If 0<n<1, the shaped reward will be set to 0 for the rest of the episode with probability n at each time step.
        """
        config = self.pseudo_reward_config

        if config is None:
            return None

        reward_type = config['type'].lower()
        all_dests = self.objects # List of all possible destinations

        if reward_type == 'subtask': # Reward for subtasks. There is only one task here, so this is just the regular reward signal.
            obj = self.env.carrying
            if obj is not None:
                if self._is_target_obj(obj):
                    return 1.0
                else:
                    return -1.0
            return 0.0

        elif reward_type == 'inverse distance': # Distance-based reward bounded by 1
            dest = self._get_closest_dest()

            if dest is None:
                return 0 # Reward of 0 if there's no target

            distance = abs(dest.cur_pos[0] - self.env.agent_pos[0]) + abs(dest.cur_pos[1] - self.env.agent_pos[1])
            
            reward = 1 / (distance + 1)

            return reward

        elif reward_type == 'zero': # Always 0. Use to test the agent's ability to handle having a pseudo-reward signal but the signal is constant.
            return 0

        elif reward_type == 'pbrs': # Same as inverse distance, but converted to a potential-based shaped reward
            raise NotImplementedError('Potential-based shaped reward not implemented')

        elif reward_type == 'adjacent to subtask':
            dest = self._get_closest_dest()
            if dest is None:
                return 0

            assert self.env.agent_pos is not None
            if self._agent_pos_prev is None:
                return 0

            dist = abs(dest.cur_pos[0] - self.env.agent_pos[0]) + abs(dest.cur_pos[1] - self.env.agent_pos[1])
            dist_prev = abs(dest.cur_pos[0] - self._agent_pos_prev[0]) + abs(dest.cur_pos[1] - self._agent_pos_prev[1])
            all_dists = [abs(obj.cur_pos[0] - self.env.agent_pos[0]) + abs(obj.cur_pos[1] - self.env.agent_pos[1]) for obj in all_dests]
            all_dists_prev = [abs(obj.cur_pos[0] - self._agent_pos_prev[0]) + abs(obj.cur_pos[1] - self._agent_pos_prev[1]) for obj in all_dests]

            if dist == 1:
                if dist_prev != 1:
                    return 1
                else:
                    return 0
            elif min(all_dists) == 1 and min(all_dists_prev) != 1:
                return -1
            else:
                return 0

        raise ValueError(f'Unknown reward type: {reward_type}')

    def _potential(self) -> float:
        dest = self._get_closest_dest()
        if dest is None:
            return 0
        dist = abs(dest.cur_pos - self.env.agent_pos[0]) + abs(dest.cur_pos - self.env.agent_pos[1])
        max_size = self.env.width + self.env.height
        return -dist / max_size

    @property
    def description(self):
        obj_names = [f'{obj.color} {obj.type}' for obj in self.target_objs]
        if len(obj_names) > 1:
            obj_names = ', '.join(obj_names[:-1]) + ' or ' + obj_names[-1]
        else:
            obj_names = obj_names[0]
        return f'Pick up the {obj_names}'


class AlternatingTasks(Task):
    """
    Cycle through each of the provided tasks. If a trial is completed successfully (i.e. with a positive reward), then we move on to the next task. The first task is chosen at random from the list of tasks.
    """
    def __init__(self,
                 env: 'MultiRoomEnv_v2',
                 rng: np.random.RandomState,
                 tasks: Sequence[Task]):
        self.env = env
        self.rng = rng
        self.tasks = tasks
    def reset(self):
        self.current_task_idx = self.rng.choice(len(self.tasks))
        for t in self.tasks:
            t.reset()
    def step(self) -> TaskStepReturn:
        output = self.tasks[self.current_task_idx].step()
        if output['trial_completed'] and output['reward'] > 0:
            self.current_task_idx = (self.current_task_idx+1) % len(self.tasks)
        return output
    @property
    def description(self):
        return self.tasks[self.current_task_idx].description


class TaskWrapper(Task):
    def __init__(self, task, rng: np.random.RandomState):
        self.task = task
        self.rng = rng
    def reset(self):
        self.task.reset()
    def step(self):
        self.task.step()
    @property
    def description(self) -> str:
        return self.task.description
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        return getattr(self.task, attr)


class RandomResetWrapper(TaskWrapper):
    def __init__(self, task, rng: np.random.RandomState, prob: float):
        """ Randomly reset the environment with probability `prob` at the end of each trial. """
        super().__init__(task, rng)
        self.prob = prob
    def step(self):
        output = self.task.step()
        if output['trial_completed'] and self.rng.uniform() < self.prob:
            self.task.reset()
        return output


def init_task(env, rng, config) -> Task:
    task_mapping = {
        'none': NoTask,
        'fetch': FetchTask,
    }
    if config['task'] in task_mapping:
        task = task_mapping[config['task']](env, rng, **config['args'])
    elif config['task'] == 'alternating':
        task = AlternatingTasks(
                env, rng,
                [init_task(env, rng, c) for c in config['args']['tasks']]
        )
    else:
        raise ValueError(f'Unknown task: {config["task"]}')

    for wrapper_config in config.get('wrappers', []):
        wrapper_mapping = {
            'random_reset': RandomResetWrapper,
        }
        if wrapper_config['type'] in wrapper_mapping:
            task = wrapper_mapping[wrapper_config['type']](task, rng, **wrapper_config['args'])
        else:
            raise ValueError(f'Unknown task wrapper: {wrapper_config["type"]}')
    return task


##################################################
# Environment
##################################################


class MultiRoomEnv_v2(MiniGridEnv):
    """
    Same as v1, but reorganized so that the task-specific code is separate from the environment code.

    Modified object placement code so obstructions are checked on objects too, not just doorways.
    """

    def __init__(self,
        # Room parameters
        min_num_rooms,
        max_num_rooms,
        min_room_size=5,
        max_room_size=10,
        door_prob=0.5,
        # Objects
        num_objs: int = 0,
        num_obj_types: int = 2,
        num_obj_colors: int = 6,
        unique_objs: bool = True,
        # Other environment stuff
        num_goals: int = 0,
        num_lava: int = 0,
        # Task parameters
        task_config: dict = {},
        num_trials=100,
        max_steps_multiplier: float = 1.,
        pseudo_reward_config: dict = {},
        seed = None,
        render_mode = 'rgb_array',
    ):
        """
        Args:
            min_num_rooms (int): minimum number of rooms
            max_num_rooms (int): maximum number of rooms
            min_room_size (int): minimum size of a room. The size includes the walls, so a room of size 3 in one dimension has one occupiable square in that dimension.
            max_room_size (int): maximum size of a room
            door_prob (float): probability of a door being placed in the opening between two rooms.
            num_objs (int): number of objects to place in the environment
            num_obj_types (int): number of object types. Available types are 'key' and 'ball', so max value is 2.
            num_obj_colors (int): number of object colors. Max value is 6.
            unique_objs (bool): if True, then each object will be unique. If False, then there may be multiple objects of the same type and color.
            num_goals (int): number of goals to place in the environment. Goals are visually represented by a green square.
            num_lava (int): number of lava squares to place in the environment.
            task_config (dict): configuration for the task. See the documentation for the Task class for more details.
            num_trials (int): number of trials per episode. A trial ends upon reaching the goal state or picking up an object. If set to -1, the environment will run indefinitely.
            max_steps_multiplier (float): maximum number of steps per episode is max_steps_multiplier * total_room_size * num_trials. If set to -1, the environment will run indefinitely.
            pseudo_reward_config (dict): configuration for the pseudo-reward signal. It can be used to specify transformations to apply to the pseudo-reward.
                Tasks output a pseudo-reward on each step, which is summed up and returned by the environment in the observation dict. If this is set to None, the pseudo-reward will be omitted from the observation. If set to an empty dict, the pseudo-reward will simply be summed up with no transformations applied.
            seed (int): random seed.
        """
        assert min_num_rooms > 0
        assert max_num_rooms >= min_num_rooms
        assert max_room_size >= 4

        self.min_num_rooms = min_num_rooms
        self.max_num_rooms = max_num_rooms
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size
        self.door_prob = door_prob

        self.num_objs = num_objs
        self.num_obj_types = num_obj_types
        self.num_obj_colors = num_obj_colors
        self.unique_objs = unique_objs

        self.num_goals = num_goals
        self.num_lava = num_lava

        self.num_trials = num_trials
        self.max_steps_multiplier = max_steps_multiplier
        self.pseudo_reward_config = pseudo_reward_config

        self.rooms = []

        self.np_random, seed = init_rng(seed)

        super(MultiRoomEnv_v2, self).__init__(
            grid_size=25,
            max_steps=self.max_num_rooms * 20 * self.num_trials,
            see_through_walls = False,
            mission_space = MissionSpace(mission_func = lambda: 'Do whatever'),
            render_mode = render_mode,
        )

        assert isinstance(self.observation_space, gymnasium.spaces.Dict)
        if pseudo_reward_config is not None:
            self.observation_space = gymnasium.spaces.Dict({
                **self.observation_space.spaces,
                'pseudo_reward': gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            })
            self.pseudo_reward_noise = RewardNoise(
                    *pseudo_reward_config.get('noise', tuple()))
            self.pseudo_reward_delay = RewardDelay(
                    *pseudo_reward_config.get('delay', tuple()))
            self.pseudo_reward_delayed_start = RewardDelayedStart(
                    *pseudo_reward_config.get('delayed_start', tuple()))
            self.pseudo_reward_transforms = [
                self.pseudo_reward_delay,
                self.pseudo_reward_noise,
                self.pseudo_reward_delayed_start,
            ]
        else:
            self.pseudo_reward_transforms = []

        self.task = init_task(self, self.np_random, task_config)

        self._agent_pos_prev = None

    def _gen_grid(self, width, height):
        """ Called by the MiniGridEnv reset() method. """
        room_list = []
        self.rooms = room_list

        # Choose a random number of rooms to generate
        num_rooms = self._rand_int(self.min_num_rooms, self.max_num_rooms+1)

        # Create first room
        room_height = self._rand_int(self.min_room_size, self.max_room_size+1)
        room_width = self._rand_int(self.min_room_size, self.max_room_size+1)
        top = self._rand_int(1, height - room_height - 1)
        left = self._rand_int(1, width - room_width - 1)
        room_list.append(Room(top, top + room_height - 1, left, left + room_width - 1))

        # Keep a list of where new rooms can potentially be placed and remove them from the list when a new room is added there.
        # List consists of a room index and the direction relative to that room.
        new_room_openings = [ (0, 'left'), (0, 'right'), (0, 'top'), (0, 'bottom') ]
        while len(room_list) < num_rooms:
            if len(new_room_openings) == 0:
                break

            # Choose a random place to connect the new room to
            r = self._rand_int(0, len(new_room_openings))
            starting_room_index, wall = new_room_openings[r]

            temp_room = self._generate_room(
                    room_list,
                    idx = starting_room_index,
                    wall = wall,
                    min_size = self.min_room_size,
                    max_size = self.max_room_size,
                    width = width,
                    height = height,
            )
            if temp_room is not None:
                room_list.append(temp_room)
                new_room_openings.append((len(room_list)-1, 'left'))
                new_room_openings.append((len(room_list)-1, 'right'))
                new_room_openings.append((len(room_list)-1, 'top'))
                new_room_openings.append((len(room_list)-1, 'bottom'))
            else:
                new_room_openings.remove(new_room_openings[r])

        self.grid = minigrid.core.grid.Grid(width, height)
        self.doors = []
        wall = minigrid.core.world_object.Wall()
        self.wall = wall

        for room in room_list:
            # Look for overlapping walls
            overlapping_walls = {
                'top': [],
                'bottom': [],
                'left': [],
                'right': [],
            }
            for i in range(room.left + 1, room.right):
                if self.grid.get(i,room.top) == wall and self.grid.get(i,room.top+1) is None and self.grid.get(i,room.top-1) is None:
                    overlapping_walls['top'].append((i,room.top))
                if self.grid.get(i,room.bottom) == wall and self.grid.get(i,room.bottom+1) is None and self.grid.get(i,room.bottom-1) is None:
                    overlapping_walls['bottom'].append((i,room.bottom))
            for j in range(room.top + 1, room.bottom):
                if self.grid.get(room.left,j) == wall and self.grid.get(room.left+1,j) is None and self.grid.get(room.left-1,j) is None:
                    overlapping_walls['left'].append((room.left,j))
                if self.grid.get(room.right,j) == wall and self.grid.get(room.right+1,j) is None and self.grid.get(room.right-1,j) is None:
                    overlapping_walls['right'].append((room.right,j))

            # Create room
            # Top wall
            for i in range(room.left, room.right + 1):
                self.grid.set(i, room.top, wall)
            # Bottom wall
            for i in range(room.left, room.right + 1):
                self.grid.set(i, room.bottom, wall)
            # Left wall
            for i in range(room.top, room.bottom + 1):
                self.grid.set(room.left, i, wall)
            # Right wall
            for i in range(room.top, room.bottom + 1):
                self.grid.set(room.right, i, wall)

            # Create doorways between rooms
            for ow in overlapping_walls.values():
                if len(ow) == 0:
                    continue
                opening = self._rand_elem(ow)
                if self.np_random.uniform() > self.door_prob:
                    self.grid.set(opening[0], opening[1], None)
                else:
                    door = minigrid.core.world_object.Door(
                        color = self._rand_elem(minigrid.core.constants.COLOR_NAMES)
                    )
                    self.grid.set(opening[0], opening[1], door)
                    self.doors.append(door)
                # Store the doorways with the room object
                for x in self.rooms:
                    x.doorways.append(opening)

        self._init_agent()

        assert isinstance(self.observation_space, gymnasium.spaces.Dict)
        self.mission = self.observation_space.spaces['mission'].sample()

        # Set max steps
        total_room_sizes = sum([room.height * room.width for room in room_list])
        self.max_steps = int(total_room_sizes * self.num_trials * self.max_steps_multiplier)
        if self.max_steps < 0:
            self.max_steps = float('inf')

    def _init_objects(self, num_objs, num_obj_types, num_obj_colors, unique_objs, num_goals, num_lava):
        # Goals and lava
        # Generate these first because they don't move, so they need to be positioned unobstructively
        self.goals = [
            minigrid.core.world_object.Goal()
            for _ in range(num_goals)
        ]
        self.lava = [
            minigrid.core.world_object.Lava()
            for _ in range(num_lava)
        ]
        for x in self.goals+self.lava:
            self.place_obj(x, unobstructive=True)

        # Objects that can be picked up
        types = ['key', 'ball'][:num_obj_types]
        colors = minigrid.core.constants.COLOR_NAMES[:num_obj_colors]

        type_color_pairs = [(t,c) for t in types for c in colors]

        objs = []

        # For each object to be generated
        while len(objs) < num_objs:
            obj_type, obj_color = self._rand_elem(type_color_pairs)
            if unique_objs:
                type_color_pairs.remove((obj_type, obj_color))

            if obj_type == 'key':
                obj = minigrid.core.world_object.Key(obj_color)
            elif obj_type == 'ball':
                obj = minigrid.core.world_object.Ball(obj_color)
            else:
                raise ValueError(f'Unknown object type: {obj_type}')

            self.place_obj(obj, unobstructive=False)
            objs.append(obj)

        self.objects = objs

    def _init_agent(self):
        # Randomize the player start position and orientation
        self.agent_pos = self._rand_space()
        self.agent_dir = self._rand_int(0, 4)

    def _rand_space(self):
        """ Find and return the coordinates of a random empty space in the grid """

        room_indices = list(range(len(self.rooms)))
        self.np_random.shuffle(room_indices) # type: ignore

        for r in room_indices:
            # Choose a random room
            room = self.rooms[r]

            # List all spaces in the room
            spaces = [
                (x, y)
                for x in range(room.left+1, room.right)
                for y in range(room.top+1, room.bottom)
            ]
            self.np_random.shuffle(spaces) # type: ignore

            # Choose a random location in the room
            for x, y in spaces:
                # Check if the location is empty
                if self.grid.get(x, y) is not None:
                    continue
                # Check if the agent is here
                if np.array_equal((x,y), self.agent_pos):
                    continue
                return np.array([x, y])

        raise Exception('Could not find a random empty space')

    def _rand_space_unobstructive(self):
        """ Find and return the coordinates of a random empty space in the grid.
        This space is chosen from a set of spaces that would not obstruct access to other parts of the environment if an object were to be placed there.
        We define a space as accessible if there exists an empty space adjacent to it and there exists a path between that space and any other empty space in the environment without passing over another object.
        """

        room_indices = list(range(len(self.rooms)))
        self.np_random.shuffle(room_indices) # type: ignore

        for r in room_indices:
            # Choose a random room
            room = self.rooms[r]

            # List all spaces in the room
            spaces = [
                (x, y)
                for x in range(room.left+1, room.right)
                for y in range(room.top+1, room.bottom)
            ]
            self.np_random.shuffle(spaces) # type: ignore

            def is_obstructive(pos):
                # Return True if placing an object in position `pos` would be an obstruction.
                # A space is reachable if the agent can be adjacent to it or in the space
                # Only need to check the room that the object is placed in
                reachable = {s:False for s in spaces+room.doorways}
                def foo(x,y):
                    # Not a space we care about
                    if (x,y) not in reachable:
                        return
                    # Space has already been processed
                    if reachable[(x,y)]:
                        return
                    reachable[(x,y)] = True
                    # Check if there's an object in this space
                    obj = self.grid.get(x,y)
                    occupied = (obj is not None and not isinstance(obj, minigrid.core.world_object.Door)) or pos == (x,y)
                    # If it's occupied, then the adjacent spaces will not be reachable from here.
                    if occupied:
                        return
                    # Otherwise, we'll check the adjacent spaces
                    foo(x-1,y)
                    foo(x+1,y)
                    foo(x,y+1)
                    foo(x,y-1)
                # Find first unoccupied space
                if len(room.doorways) > 0:
                    foo(*room.doorways[0])
                else:
                    x,y = None,None
                    for x,y in spaces:
                        if self.grid.get(x, y) is not None:
                            continue
                        if pos == (x,y):
                            continue
                        break
                    if x is None or y is None:
                        return True
                    foo(x,y)
                return not all(reachable.values())

            # Choose a random location in the room
            for x, y in spaces:
                # Check if the location is empty
                if self.grid.get(x, y) is not None:
                    continue
                # Check if the agent is here
                if np.array_equal((x,y), self.agent_pos):
                    continue
                # Check if it blocks a doorway
                obstructive = False
                for d in [[0,1],[1,0],[0,-1],[-1,0]]:
                    if self.grid.get(x+d[0], y+d[1]) is self.wall:
                        continue
                    c1 = [d[1]+d[0],d[1]+d[0]]
                    c2 = [-d[1]+d[0],d[1]-d[0]]
                    if self.grid.get(x+c1[0], y+c1[1]) is self.wall and self.grid.get(x+c2[0], y+c2[1]) is self.wall:
                        obstructive = True
                        break

                obstructive = obstructive or is_obstructive((x,y))

                if obstructive:
                    continue

                return np.array([x, y])

        raise Exception('Could not find a random empty space')

    def place_obj(self, obj, unobstructive=False):
        if unobstructive:
            pos = self._rand_space_unobstructive()
        else:
            pos = self._rand_space()

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def _generate_room(self, rooms, idx, wall, min_size, max_size, width, height):
        starting_room = rooms[idx]
        new_room = Room(0,0,0,0)
        if wall == 'left' or wall == 'right':
            min_top = max(starting_room.top - max_size + 3, 0)
            max_top = starting_room.bottom - 2
            min_bottom = starting_room.top + 2
            max_bottom = starting_room.bottom + max_size - 3
            if wall == 'left':
                #new_room.right = starting_room.left
                min_right = starting_room.left
                max_right = starting_room.left
                min_left = max(starting_room.left - max_size + 1, 0)
                max_left = starting_room.left - min_size + 1
            else:
                #new_room.left = starting_room.right
                min_left = starting_room.right
                max_left = starting_room.right
                min_right = starting_room.right + min_size - 1
                max_right = starting_room.right + max_size - 1
        else:
            min_left = max(starting_room.left - max_size + 3, 0)
            max_left = starting_room.right - 2
            min_right = starting_room.left + 2
            max_right = starting_room.right + max_size - 3
            if wall == 'top':
                #new_room.bottom = starting_room.top
                min_bottom = starting_room.top
                max_bottom = starting_room.top
                min_top = max(starting_room.top - max_size + 1, 0)
                max_top = starting_room.top - min_size + 1
            else:
                #new_room.top = starting_room.bottom
                min_top = starting_room.bottom
                max_top = starting_room.bottom
                min_bottom = starting_room.bottom + min_size - 1
                max_bottom = starting_room.bottom + max_size - 1
        possible_rooms = [
            (t,b,l,r)
            for t in range(min_top, max_top + 1)
            for b in range(max(min_bottom,t+min_size-1), min(max_bottom + 1, t+max_size))
            for l in range(min_left, max_left + 1)
            for r in range(max(min_right,l+min_size-1), min(max_right + 1, l+max_size))
        ]
        self.np_random.shuffle(possible_rooms) # type: ignore
        for room in possible_rooms:
            new_room.top = room[0]
            new_room.bottom = room[1]
            new_room.left = room[2]
            new_room.right = room[3]
            if room_is_valid(rooms, new_room, width, height):
                return new_room
        return None

    @property
    def goal_str(self):
        return self.task.description

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        self._init_objects(
            num_objs = self.num_objs,
            num_obj_types = self.num_obj_types,
            num_obj_colors = self.num_obj_colors,
            unique_objs = self.unique_objs,
            num_goals = self.num_goals,
            num_lava = self.num_lava,
        )

        self.trial_count = 0
        self._agent_pos_prev = None
        info['reward'] = 0

        self.task.reset()
        for transform in self.pseudo_reward_transforms:
            transform.reset()
        if self.pseudo_reward_config is not None:
            obs['pseudo_reward'] = np.array([0], dtype=np.float32)

        return obs, info

    def step(self, action):
        self._agent_pos_prev = self.agent_pos

        obs, reward, terminated, truncated, info = super().step(action)

        # Tasks
        task_output = self.task.step()
        reward = task_output['reward']
        if task_output['trial_completed']:
            self.trial_count += 1
            # Inform shaped reward transforms that a trial was completed
            for transform in self.pseudo_reward_transforms:
                transform.trial_finished()
        if self.pseudo_reward_config is not None:
            pseudo_reward = task_output['pseudo_reward']
            for transform in self.pseudo_reward_transforms:
                pseudo_reward = transform(pseudo_reward)
            obs['pseudo_reward'] = np.array([pseudo_reward], dtype=np.float32)

        # Reset objects the agent picked up
        if self.carrying:
            # Place the object back in the environment
            self.place_obj(self.carrying)
            # Remove the object from the agent's hand
            self.carrying = None

        # If the agent steps on a goal square, teleport the agent to a random empty square
        curr_cell = self.grid.get(*self.agent_pos) # type: ignore
        if curr_cell != None and hasattr(curr_cell,'rewards') and hasattr(curr_cell,'probs'):
            # Teleport the agent to a random location
            self._init_agent()

        # Terminate if trial count is exceeded
        if self.num_trials > 0 and self.trial_count >= self.num_trials:
            terminated = True
            self.trial_count = 0

        return obs, reward, terminated, truncated, info


if __name__ == '__main__':
    env = gymnasium.make('MiniGrid-MultiRoom-v2',
            min_num_rooms=4,
            max_num_rooms=6,
            min_room_size=4,
            max_room_size=10,
            #min_num_rooms=1,
            #max_num_rooms=2,
            #min_room_size=4,
            #max_room_size=5,
            # Objects
            #num_objs = 6,
            num_objs = 6*2,
            num_obj_types = 2,
            num_obj_colors = 6,
            unique_objs = True,
            # Other environment stuff
            num_goals = 1,
            num_lava = 1,

            seed=0,
            render_mode='human'
    )
    env.reset()
    env.render()
    breakpoint()

