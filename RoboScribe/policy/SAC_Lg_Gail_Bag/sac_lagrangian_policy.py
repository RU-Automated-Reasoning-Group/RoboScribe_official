from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BaseModel

import torch
import torch.nn as nn
import numpy as np

import pdb

# SAC policy
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

# Discriminator
class Discriminator(BaseModel):
    def __init__(self,
                observation_space,
                action_space,
                net_arch,
                only_obs = False,
                features_extractor = None,
                share_features_extractor = True,
                normalize_images = True,
                optimizer_class = torch.optim.Adam,
                optimizer_kwargs = None):
        
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images
        )

        # init
        self.net_arch = net_arch
        self.only_obs = only_obs
        if only_obs:
            # self.inp_dim = observation_space.shape[0]
            self.inp_dim = features_extractor.features_dim
        else:
            # self.inp_dim = observation_space.shape[0] + action_space.shape[0]
            self.inp_dim = features_extractor.features_dim + action_space.shape[0]

        # get network
        self.d = []
        for in_dim, out_dim in zip([self.inp_dim]+net_arch, net_arch+[1]):
            self.d.append(nn.Linear(in_dim, out_dim))
            self.d.append(nn.ReLU(inplace=True))
        self.d.pop(-1)
        self.d = nn.Sequential(*self.d)
        self.features_extractor = features_extractor
        self.share_features_extractor = share_features_extractor
        self.sigmoid = nn.Sigmoid()

        # get optimizer
        if self.share_features_extractor and self.features_extractor is not None:
            d_parameters = [param for name, param in self.d.named_parameters() if "features_extractor" not in name]
        else:
            d_parameters = list(self.d.parameters())
        self.d_optim = optimizer_class(params=d_parameters, **optimizer_kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, observation, actions, no_last_act=False):
        if no_last_act:
            return self.get_logits(observation, actions)
        return self.sigmoid(self.get_logits(observation, actions))
    
    def get_logits(self, observation, actions):
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(observation, self.features_extractor)
        if self.only_obs:
            d_inp = features
        else:
            d_inp = torch.cat([features, actions], dim=1)
        return self.d(d_inp)

# expert data
class ExpertData:
    def __init__(self, expert_data, device):
        # store
        self.state_data = expert_data['state']
        self.action_data = expert_data['action']
        self.done_data = expert_data['done'] 
        self.device = device
        # calculate distance for regression
        self.dist_data = np.zeros(self.state_data.shape[0])
        end_ids = np.where(expert_data['done'])[0]
        last_id = 0
        for end_id in end_ids:
            for cur_id in range(last_id, end_id+1):
                self.dist_data[cur_id] = end_id - cur_id
            last_id = end_id+1
        self.dist_data = np.power(0.95, self.dist_data)

    def sample(self, batch_size, only_end=False, with_dist=False):
        # sample batch size of data
        batch_size = min(batch_size, self.state_data.shape[0])
        if only_end:
            valid_idxs = np.where(self.done_data)[0]
            if valid_idxs.shape[0] < batch_size:
                sample_idxs = valid_idxs
            else:
                sample_idxs = np.random.choice(valid_idxs, size=batch_size, replace=False)
        else:
            sample_idxs = np.random.choice(np.arange(self.state_data.shape[0]), size=batch_size, replace=False)
        sample_states = self.state_data[sample_idxs]

        # make torch
        return_list = [torch.tensor(sample_states, dtype=torch.float32).to(self.device)]
        if self.action_data is None:
            return_list.append(None)
        else:
            sample_actions = self.action_data[sample_idxs]
            return_list.append(torch.tensor(sample_actions, dtype=torch.float32).to(self.device))

        if with_dist:
            pdb.set_trace()
            sample_dist = self.dist_data[sample_idxs]
            return_list.append(torch.tensor(sample_dist, dtype=torch.float32).to(self.device))
        
        return return_list

    # only without dist for now
    def add(self, state, action, done):
        self.state_data = np.concatenate([self.state_data, state], axis=0)
        self.action_data = np.concatenate([self.action_data, action], axis=0)
        self.done_data = np.concatenate([self.done_data, done], axis=0)

    def get_len(self):
        return self.state_data.shape[0]

# Bag of SAC Lag
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

# Bag of Discriminator
class DiscriminatorBag(nn.Module):
    def __init__(self, d_dict):
        super().__init__()
        self.d_dict = d_dict

    # load discriminator
    def load_d(self, d_dict):
        # load each d
        for d_id in self.d_dict:
            # init
            if str(d_id) not in self.d_dict:
                continue
            load_d = d_dict[d_id]
            self.d_dict[str(d_id)].load_state_dict(load_d.state_dict())


MlpPolicy = SACLagPolicy
CNNPolicy = None
MultiInputPolicy = None