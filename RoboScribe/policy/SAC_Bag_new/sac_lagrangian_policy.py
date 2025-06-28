from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import Schedule
import numpy as np
import torch
import torch.nn as nn

import pdb

class SACPolicyBag(nn.Module):
    def __init__(self, policy_dict, env_policy_used):
        super().__init__()

        self.policy_dict = policy_dict
        self.env_policy_used = env_policy_used

    def set_training_mode(self, mode: bool) -> None:
        for policy_id in self.policy_dict:
            self.policy_dict[policy_id].set_training_mode(mode)

    def predict(self,
                observation,
                state = None,
                episode_start = None,
                deterministic = False,
                policy_id = None):
        if policy_id is None:
            assert observation.shape[0] == len(self.env_policy_used)
            actions = []
            for num_id in self.env_policy_used:
                cur_obs = observation[num_id: num_id+1]
                cur_state = state[num_id: num_id+1] if state is not None else None
                cur_episode_start = episode_start[num_id] if episode_start is not None else None
                actions.append(self.policy_dict[str(self.env_policy_used[num_id])].predict(cur_obs, cur_state, cur_episode_start, deterministic)[0])
            return np.concatenate(actions, axis=0), None
        else:
            return self.policy_dict[str(policy_id)].predict(observation, state, episode_start, deterministic)

    # load policies
    def load_policy(self, policy_dict):
        # load each policy
        for policy_id in policy_dict:
            # init
            if str(policy_id) not in self.policy_dict:
                continue
            load_policy = policy_dict[policy_id]
            # load policy
            self.policy_dict[str(policy_id)].actor.load_state_dict(load_policy.actor.state_dict())
            self.policy_dict[str(policy_id)].critic.load_state_dict(load_policy.critic.state_dict())
            self.policy_dict[str(policy_id)].critic_target.load_state_dict(load_policy.critic_target.state_dict())

    def scale_action(self, action):
        return self.policy_dict[str(0)].scale_action(action)
    
    def unscale_action(self, action):
        return self.policy_dict[str(0)].unscale_action(action)


MlpPolicy = SACPolicy
CNNPolicy = None
MultiInputPolicy = None