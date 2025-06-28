from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import Schedule
import numpy as np
import torch
import torch.nn as nn

import pdb

class SACLagPolicy(SACPolicy):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # init for SAC
        super().__init__(
            *args, **kwargs
        )

    # rewrite build
    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)

        # critic for lagrangian
        if self.share_features_extractor:
            self.critic_cost = self.make_critic(features_extractor=self.actor.features_extractor)
            critic_cost_parameters = [param for name, param in self.critic_cost.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic_cost = self.make_critic(features_extractor=None)
            critic_cost_parameters = list(self.critic_cost.parameters())

        # critic target for lagrangian
        self.critic_cost_target = self.make_critic(features_extractor=None)
        self.critic_cost_target.load_state_dict(self.critic_cost.state_dict())

        self.critic_cost.optimizer = self.optimizer_class(
            critic_cost_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_cost_target.set_training_mode(False)

    # rewrite to inclue new critic
    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)

        # add
        self.critic_cost.set_training_mode(mode)

class SACLagPolicyBag(nn.Module):
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
            self.policy_dict[str(policy_id)].critic_cost.load_state_dict(load_policy.critic_cost.state_dict())
            self.policy_dict[str(policy_id)].critic_cost_target.load_state_dict(load_policy.critic_cost_target.state_dict())

    def scale_action(self, action):
        return self.policy_dict[str(0)].scale_action(action)
    
    def unscale_action(self, action):
        return self.policy_dict[str(0)].unscale_action(action)


MlpPolicy = SACLagPolicy
CNNPolicy = None
MultiInputPolicy = None