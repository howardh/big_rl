import os
from typing import Generator, NamedTuple

TOTAL_NUM_TASKS = 9

class ResultsDir(NamedTuple):
    default: str
    shuffled_obs: str
    shuffled_action: str
    shuffled_obs_and_action: str
    occluded_obs_100: str
    occluded_obs_action_reward_100: str

    @classmethod
    def from_results_dir(cls, results_dir: str):
        return cls(
            default=os.path.join(results_dir, 'eval_results'),
            shuffled_obs=os.path.join(results_dir, 'eval_shuffled_obs_results'),
            shuffled_action=os.path.join(results_dir, 'eval_shuffled_action_results'),
            shuffled_obs_and_action=os.path.join(results_dir, 'eval_shuffled_obs_and_action_results'),
            occluded_obs_100=os.path.join(results_dir, 'eval_occluded_obs_100_results'),
            occluded_obs_action_reward_100=os.path.join(results_dir, 'eval_occluded_obs_action_reward_100_results'),
        )
