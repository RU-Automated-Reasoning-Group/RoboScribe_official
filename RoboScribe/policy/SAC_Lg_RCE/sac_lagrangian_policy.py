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
            self.inp_dim = observation_space.shape[0]
        else:
            self.inp_dim = observation_space.shape[0] + action_space.shape[0]

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

    def forward(self, observation, actions):
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
        self.state_data = expert_data['state']
        self.action_data = expert_data['action']
        self.done_data = expert_data['done']
        self.device = device

    def sample(self, batch_size, only_end=False):
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
        if self.action_data is None:
            sample_actions = None
            return torch.tensor(sample_states, dtype=torch.float32).to(self.device), None
        else:
            sample_actions = self.action_data[sample_idxs]
            return torch.tensor(sample_states, dtype=torch.float32).to(self.device), torch.tensor(sample_actions, dtype=torch.float32).to(self.device)


MlpPolicy = SACLagPolicy
CNNPolicy = None
MultiInputPolicy = None