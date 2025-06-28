from typing import Optional, Union, Dict, Any, List, Type

import torch
import torch.nn as nn
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from torch.distributions.normal import Normal
import copy

TensorDict = Dict[str, torch.Tensor]
PyTorchObs = Union[torch.Tensor, TensorDict]

from policy.rce_sb3.models import ContinuousCriticCls

import pdb

def glorot_init(p):
    if isinstance(p, nn.Linear):
        nn.init.xavier_normal_(p.weight.data, gain=1.)
        nn.init.zeros_(p.bias)


class tanh_gaussian_actor(nn.Module):
    def __init__(self, input_dims, action_dims, net_arch, log_std_min, log_std_max):
        super(tanh_gaussian_actor, self).__init__()
        # fc layers
        self.fc_layers = [nn.Linear(input_dims, net_arch[0]), nn.ReLU()]
        for inp_dim, outp_dim in zip(net_arch[:-1], net_arch[1:]):
            self.fc_layers.append(nn.Linear(inp_dim, outp_dim))
            self.fc_layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*self.fc_layers)
        # output layers
        self.mean = nn.Linear(net_arch[-1], action_dims)
        self.log_std = nn.Linear(net_arch[-1], action_dims)
        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # init the networks
        self.apply(glorot_init)

    def forward(self, obs):
        if obs.dtype != torch.float32:
            obs = obs.type(torch.float32)
        x = self.fc_layers(obs)
        mean = self.mean(x)
        log_std = self.log_std(x)
        # clamp the log std
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        # the reparameterization trick
        # return mean and std
        return (mean, torch.exp(log_std))


class Critic(nn.Module):
    """
    construct a classifier C(s,a) -> [0,1]
    """

    def __init__(self, obs_dim, action_dim, net_arch):
        super(Critic, self).__init__()
        # fc layers
        self.fc_layers = [nn.Linear(obs_dim+action_dim, net_arch[0]), nn.ReLU()]
        for inp_dim, outp_dim in zip(net_arch[:-1], net_arch[1:]):
            self.fc_layers.append(nn.Linear(inp_dim, outp_dim))
            self.fc_layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*self.fc_layers)
        # output layers
        self.q = nn.Linear(net_arch[-1], 1)

        self.apply(glorot_init)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1).type(torch.float32)
        x = self.fc_layers(x)
        q = self.q(x)
        q = torch.sigmoid(q)
        return q

class RCEPolicy(BasePolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        log_std_min: float = -2.,
        log_std_max: float=-20.,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        use_sde: bool = False
    ):
        # init for SAC
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )
        self.share_features_extractor = share_features_extractor

        # create actor
        self.actor = tanh_gaussian_actor(input_dims=observation_space.shape[0], 
                                         action_dims=action_space.shape[0], 
                                         net_arch=net_arch, 
                                         log_std_min=log_std_min, 
                                         log_std_max=log_std_max).to(self.device)
        # create critic 1 and 2
        self.critic_1 = Critic(obs_dim=observation_space.shape[0], 
                             action_dim=action_space.shape[0], 
                             net_arch=net_arch).to(self.device)
        self.critic_2 = Critic(obs_dim=observation_space.shape[0], 
                             action_dim=action_space.shape[0], 
                             net_arch=net_arch).to(self.device)
        # create critic target
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        action, pre_tanh_action =  self.select_action(observation, not deterministic)

        return action.detach()
    
    def select_action(self, s, rsample=True, return_action_log_probs=False):
        mean, std = self.actor(s)
        if rsample:
            pre_tanh_action = mean + torch.randn(mean.size()).to(self.device) * std
            action = torch.tanh(pre_tanh_action)
        else:
            pre_tanh_action = Normal(mean, std).sample()
            action = torch.tanh(pre_tanh_action).detach()

        return action, pre_tanh_action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        return data


MlpPolicy = RCEPolicy
CNNPolicy = None
MultiInputPolicy = None