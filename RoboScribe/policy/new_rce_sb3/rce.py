from typing import Optional, Union, Dict, Any, Type, TypeVar, Tuple, ClassVar, List

import torch
import numpy as np
from policy.rce.replaybuffer import UniformReplayBuffer, N_step_traj
from policy.new_rce_sb3.rce_policy import tanh_gaussian_actor, Critic, RCEPolicy, MlpPolicy, CNNPolicy, MultiInputPolicy
from torch.distributions.normal import Normal
from gymnasium import spaces
import copy
import os
import wandb
from tqdm import tqdm

import time

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy

import pdb

SelfRCE = TypeVar("SelfRCE", bound="RCE")

class RCE(OffPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": RCEPolicy,
        "MultiInputPolicy": RCEPolicy,
    }
    policy: RCEPolicy
    actor: tanh_gaussian_actor
    critic: Critic
    critic_target: Critic


    def __init__(self,
                policy: Union[str, Type[RCEPolicy]],
                env,
                expert_examples_buffer=None,
                future_step=10,
                learning_rate = 3e-4,
                buffer_size: int = 1_000_000,  # 1e6
                learning_starts: int = 100,
                batch_size: int = 256,
                tau: float = 0.005,
                gamma: float = 0.99,
                train_freq: Union[int, Tuple[int, str]] = 1,
                gradient_steps: int = 1,
                action_noise = None,
                replay_buffer_class = None,
                replay_buffer_kwargs = None,
                optimize_memory_usage: bool = False,
                ent_coef: Union[str, float] = "auto",
                target_update_interval: int = 1,
                target_entropy: Union[str, float] = "auto",
                use_sde: bool = False,
                sde_sample_freq: int = -1,
                use_sde_at_warmup: bool = False,
                stats_window_size: int = 100,
                tensorboard_log: Optional[str] = None,
                policy_kwargs = None,
                verbose: int = 0,
                seed: Optional[int] = None,
                device: Union[torch.device, str] = "auto",
                _init_setup_model: bool = True,
        ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.max_action = env.action_space.shape[0]
        self.expert_buffer = expert_examples_buffer
        self.future_step = future_step
        self.critic_criterion = torch.nn.MSELoss(reduction='none')
        self.polyak = 1 - self.tau

        # entropy regularizer only for SAC, for RCE the entropy coef is fixed
        if type(ent_coef) == str:
            self.ent_coef = ent_coef
        else:
            self.ent_coef = float(ent_coef)

        # set up model
        self._setup_model()
        self.actor = self.policy.actor
        self.critic_1 = self.policy.critic_1
        self.critic_2 = self.policy.critic_2
        self.target_critic_1 = self.policy.target_critic_1
        self.target_critic_2 = self.policy.target_critic_2

        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_1_optim = torch.optim.Adam(self.critic_1.parameters(),
                                               lr=learning_rate)
        self.critic_2_optim = torch.optim.Adam(self.critic_2.parameters(),
                                               lr=learning_rate)

        self.global_step = 0


    def train(self, gradient_steps: int, batch_size: int = 64):
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            cur_time = time.time()

            # critic loss is "c" -> rce
            expert_transitions = self.expert_buffer.sample(batch_size)
            expert_states = self.to_tensor(expert_transitions)  # success examples

            replay_data = self.replay_buffer.sample(self.future_step, batch_size, env=self._vec_normalize_env)

            pdb.set_trace()

            # s_{t},a_{t},s_{t+1},s_{t+n},d_{t+n}
            states = self.to_tensor(replay_data.observations)
            actions = self.to_tensor(replay_data.actions)
            next_states = self.to_tensor(replay_data.next_observations)
            future_states = self.to_tensor(replay_data.future_observations)

            # compute the targets
            with torch.no_grad():
                next_actions, _ = self.select_action(next_states)
                target_q_1 = self.target_critic_1(next_states, next_actions)
                target_q_2 = self.target_critic_2(next_states, next_actions)

                future_actions, _ = self.select_action(future_states)
                target_q_future_1 = self.target_critic_1(future_states, future_actions)
                target_q_future_2 = self.target_critic_2(future_states, future_actions)

                gamma_n = self.gamma ** self.future_step
                target_q_1 = (target_q_1 + gamma_n * target_q_future_1) / 2.0
                target_q_2 = (target_q_2 + gamma_n * target_q_future_2) / 2.0

                target_q = torch.min(target_q_1, target_q_2)

                w = target_q / (1. - target_q)
                td_targets = self.gamma * w / (1. + self.gamma * w)

            td_targets = torch.cat([torch.ones(expert_states.shape[0], 1).to(self.device), td_targets], dim=0)
            weights = torch.cat(
                [torch.ones(expert_states.shape[0], 1).to(self.device) - self.gamma, 1. + self.gamma * w],
                dim=0)

            # compute the predictions
            expert_actions, _ = self.select_action(expert_states)
            pred_expert_1 = self.critic_1(expert_states, expert_actions)
            pred_expert_2 = self.critic_2(expert_states, expert_actions)

            pred_1 = self.critic_1(states, actions)
            pred_2 = self.critic_2(states, actions)

            pred_1 = torch.cat([pred_expert_1, pred_1], dim=0)
            pred_2 = torch.cat([pred_expert_2, pred_2], dim=0)

            critic_1_loss = (weights * self.critic_criterion(pred_1, td_targets)).mean()
            critic_2_loss = (weights * self.critic_criterion(pred_2, td_targets)).mean()

            self.critic_1_optim.zero_grad()
            critic_1_loss.backward(retain_graph=True)
            self.critic_1_optim.step()
            self.critic_2_optim.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optim.step()
            critic_losses.append((critic_1_loss.item()+critic_2_loss.item())/2.0)

            # soft update the target critic networks
            self._soft_update_target_network(self.target_critic_1, self.critic_1)
            self._soft_update_target_network(self.target_critic_2, self.critic_2)

            # update the actor
            actions_new, _, log_probs = self.select_action(states, return_action_log_probs=True)
            target_q1 = self.critic_1(states, actions_new)
            target_q2 = self.critic_2(states, actions_new)
            targets_q = torch.min(target_q1, target_q2)
            actor_loss = (self.ent_coef * log_probs - targets_q).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            actor_losses.append(actor_loss.item())

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(self.ent_coef))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def select_action(self, s, rsample=True, return_action_log_probs=False):

        mean, std = self.actor(s)
        if rsample:
            pre_tanh_action = mean + torch.randn(mean.size()).to(self.device) * std
            action = torch.tanh(pre_tanh_action)
            action.requires_grad_()
        else:
            pre_tanh_action = Normal(mean, std).sample()
            action = torch.tanh(pre_tanh_action).detach()

        if return_action_log_probs:
            actions_probs = Normal(mean, std).log_prob(pre_tanh_action) - torch.log(1 - action ** 2 + 1e-6)
            actions_probs = actions_probs.sum(dim=1, keepdim=True)
            return action, pre_tanh_action, actions_probs

        return action, pre_tanh_action

    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)

    def to_tensor(self, x, type=torch.float32):
        # return torch.tensor(x, dtype=type).to(self.device)
        return torch.tensor(x, dtype=type).to(self.device)

    def learn(
        self: SelfRCE,
        total_timesteps: int,
        callback = None,
        log_interval: int = 4,
        tb_log_name: str = "RCE",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRCE:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor_optim", "critic_1_optim", "critic_2_optim"]
        saved_pytorch_variables = []
        return state_dicts, saved_pytorch_variables