from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

from environment.fetch_custom.get_fetch_env import get_env, get_pickplace_env
from environment.general_env import GeneralEnv

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.policies import BasePolicy
from policy.SAC_Lg_Gail.sac_lagrangian_policy import MlpPolicy, CNNPolicy, MultiInputPolicy, SACLagPolicy, Discriminator, ExpertData

import math
import copy
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F

import gymnasium as gym

import pdb

class SACLagRce(SAC):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": SACLagPolicy,
        "MultiInputPolicy": SACLagPolicy,
    }
    policy: SACLagPolicy

    def __init__(
        self,
        lam_disable = False,
        exp_data = None,
        rce_kwargs = None,
        lam_kwargs = None,
        _init_setup_model = True,
        *args,
        **kwargs,
    ):
        # init for SAC
        if '_init_setup_model' in kwargs:
            del kwargs['_init_setup_model']
        super().__init__(
            _init_setup_model=False, *args, **kwargs
        )

        # init for lagrangian
        self.lam_kwargs = lam_kwargs
        self.lam_disable = lam_disable

        # init for discriminator
        self.ori_exp_data = exp_data
        if self.ori_exp_data is not None:
            self.exp_data = ExpertData(self.ori_exp_data, self.device)

        self.rce_kwargs = rce_kwargs
        
        self.dual_interval = 0
        self.cost_lim = -5e-4

        # setup model
        if _init_setup_model:
            self._setup_model()

    def _setup_lam(self) -> None:
        # init
        self.lam_init = 10.0
        self.lam_update_interval = 12
        self.lam_lr = 0.03
        # load
        if self.lam_kwargs is not None:
            if 'lam_init' in self.lam_kwargs:
                self.lam_init = self.lam_kwargs['lam_init']
            if 'lam_update_interval' in self.lam_kwargs:
                self.lam_update_interval = self.lam_kwargs['lam_update_interval']
            if 'lam_lr' in self.lam_kwargs:
                self.lam_lr = self.lam_kwargs['lam_lr']
        # setup others
        self.lam = th.tensor(self.lam_init, requires_grad=True)
        self.lam_optim = th.optim.Adam([self.lam], lr=self.lam_lr)

    def _setup_rce(self) -> None:
        # init
        self.rce_w = 1.
        self.rce_lr = 0.01
        self.rce_net_arch = [256, 256]
        self.rce_epoch = 2
        self.rce_batch = 1024
        self.bce_loss = nn.BCELoss()
        self.rce_only_obs = True
        self.rce_polyak = 1 - 0.005
        # load
        if self.rce_kwargs is not None:
            if 'rce_w' in self.rce_kwargs:
                self.rce_w = self.rce_kwargs['rce_w']
            if 'rce_lr' in self.rce_kwargs:
                self.rce_lr = self.rce_kwargs['rc_lr']
            if 'rce_net_arch' in self.rce_kwargs:
                self.rce_net_arch = self.rce_kwargs['rce_net_arch']
            if 'rce_epoch' in self.rce_kwargs:
                self.rce_epoch = self.rce_kwargs['rce_epoch']
            if 'rce_batch' in self.rce_kwargs:
                self.rce_batch = self.rce_kwargs['rce_batch']
            if 'rce_only_obs' in self.rce_kwargs:
                self.rce_only_obs = self.rce_kwargs['rce_only_obs']
            if 'rce_polyak' in self.rce_kwargs:
                self.rce_polyak = self.rce_kwargs['rce_polyak']

    def _setup_model(self) -> None:
        # setup policies
        super()._setup_model()

        # setup lambda
        self._setup_lam()
        # setup discriminator
        self._setup_rce()

        # setup discriminator (1-step)
        self.critic_rce_1 = Discriminator(self.observation_space, self.action_space, self.rce_net_arch, only_obs=self.rce_only_obs, \
                               features_extractor=self.actor.features_extractor, optimizer_kwargs={'lr': self.rce_lr}).to(self.device)
        self.critic_rce_2 = Discriminator(self.observation_space, self.action_space, self.rce_net_arch, only_obs=self.rce_only_obs, \
                               features_extractor=self.actor.features_extractor, optimizer_kwargs={'lr': self.rce_lr}).to(self.device)
        self.target_critic_rce_1 = copy.deepcopy(self.critic_rce_1).to(self.device)
        self.target_critic_rce_2 = copy.deepcopy(self.critic_rce_2).to(self.device)


        # add expert data
        if self.replay_buffer is not None and self.ori_exp_data is not None:
            self.replay_buffer.add_exp_data(self.ori_exp_data)

        # create aliases
        self._create_aliases()

    def _create_aliases(self) -> None:
        super()._create_aliases()
        self.critic_cost = self.policy.critic_cost
        self.critic_cost_target = self.policy.critic_cost_target

    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.rce_polyak) * param.data + self.rce_polyak * target_param.data)

    # rewrite to include lagrangian training (TODO)
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.critic_cost.optimizer, self.critic_rce_1.d_optim, self.critic_rce_2.d_optim]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, critic_cost_losses, lam_loss, d_loss = [], [], [], [], []
        rce_pred_val = []
        rce_pred_neg_val = []
        rce_rewards_store = []
        env_rewards_store = []
        critic_costs = []
        rce_critic_losses = []
        penalty_applied = []

        # train policy
        for gradient_step in range(gradient_steps):
            self.dual_interval += 1

            # Sample expert buffer
            expert_states, _ = self.exp_data.sample(self.rce_batch, only_end=True)

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            critic_costs.append(replay_data.costs.mean().item())

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            ############################
            # Train RCE critic first
            ############################
            with th.no_grad():
                # Select action according to policy
                next_actions, _ = self.actor.action_log_prob(replay_data.next_observations)
                target_rce_q_1 = self.target_critic_rce_1(replay_data.next_observations, next_actions)
                target_rce_q_2 = self.target_critic_rce_2(replay_data.next_observations, next_actions)

                target_rce_q = th.min(target_rce_q_1, target_rce_q_2)

                w = target_rce_q / (1. - target_rce_q)
                rce_targets = self.gamma * w / (1. + self.gamma * w)

            rce_targets = th.cat([th.ones(expert_states.shape[0], 1).to(self.device), rce_targets], dim=0)
            weights = th.cat(
                [th.ones(expert_states.shape[0], 1).to(self.device) - self.gamma, 1. + self.gamma * w],
                dim=0)

            # compute the predictions
            expert_actions, _ = self.actor.action_log_prob(expert_states)
            pred_expert_critic_1 = self.critic_rce_1(expert_states, expert_actions)
            pred_expert_critic_2 = self.critic_rce_2(expert_states, expert_actions)

            rce_pred_val.append(th.min(pred_expert_critic_1, pred_expert_critic_2).mean().detach().item())

            pred_critic_1 = self.critic_rce_1(replay_data.observations, replay_data.actions)
            pred_critic_2 = self.critic_rce_2(replay_data.observations, replay_data.actions)

            rce_pred_neg_val.append(th.min(pred_critic_1, pred_critic_2).mean().detach().item())

            pred_critic_1 = th.cat([pred_expert_critic_1, pred_critic_1], dim=0)
            pred_critic_2 = th.cat([pred_expert_critic_2, pred_critic_2], dim=0)

            rce_critic_1_loss = (weights * F.mse_loss(pred_critic_1, rce_targets)).mean()
            rce_critic_2_loss = (weights * F.mse_loss(pred_critic_2, rce_targets)).mean()

            self.critic_rce_1.d_optim.zero_grad()
            rce_critic_1_loss.backward(retain_graph=True)
            self.critic_rce_1.d_optim.step()
            self.critic_rce_2.d_optim.zero_grad()
            rce_critic_2_loss.backward()
            self.critic_rce_2.d_optim.step()
            rce_critic_losses.append((rce_critic_1_loss.item()+rce_critic_2_loss.item())/2.0)

            # soft update the target critic networks
            self._soft_update_target_network(self.target_critic_rce_1, self.critic_rce_1)
            self._soft_update_target_network(self.target_critic_rce_2, self.critic_rce_2)

            ############################
            # Train SAC critic first
            ############################
            with th.no_grad():
                # get rce predicted reward
                rce_reward_1 = self.target_critic_rce_1(replay_data.observations, replay_data.actions)
                rce_reward_2 = self.target_critic_rce_2(replay_data.observations, replay_data.actions)
                rce_reward = th.min(rce_reward_1, rce_reward_2)
                rce_rewards_store.append(rce_reward.mean().detach().item())
                env_rewards_store.append(replay_data.rewards.mean().item())

                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)

                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                # target_q_values = replay_data.rewards + self.rce_w * (rce_reward -1) + (1 - replay_data.dones) * self.gamma * next_q_values
                target_q_values = (rce_reward - 1) + (1 - replay_data.dones) * self.gamma * next_q_values

                if not self.lam_disable:
                    # Compute the next constrain values: min over all constrains targets
                    next_q_values_cost = th.cat(self.critic_cost_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values_cost, _ = th.min(next_q_values_cost, dim=1, keepdim=True)
                    # add entropy term
                    # next_q_values_cost = next_q_values_cost - ent_coef * next_log_prob.reshape(-1, 1)
                    next_q_values_cost = next_q_values_cost
                    # td erro + entropy term
                    target_q_values_cost = replay_data.costs + (1 - replay_data.dones) * self.gamma * next_q_values_cost

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]
            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if not self.lam_disable:
                # get current cost
                current_q_values_cost = self.critic_cost(replay_data.observations, replay_data.actions)
                # Compute critic cost loss
                critic_cost_loss = 0.5 * sum(F.mse_loss(current_q_cost, target_q_values_cost) for current_q_cost in current_q_values_cost)
                assert isinstance(critic_cost_loss, th.Tensor)
                critic_cost_losses.append(critic_cost_loss.item())
                # Optimize the critic cost
                self.critic_cost.optimizer.zero_grad()
                critic_cost_loss.backward()
                self.critic_cost.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            if self.lam_disable:
                penalty = 0
                penalty_applied.append(penalty)
            else:
                # calculate penalty from cost
                q_values_cost_pi = th.cat(self.critic_cost(replay_data.observations, actions_pi), dim=1)
                min_qf_cost_pi, _ = th.min(q_values_cost_pi, dim=1, keepdim=True)
                # penalty = max(self.lam.item(), 0) * min_qf_cost_pi
                penalty = F.softplus(self.lam).item() * min_qf_cost_pi
                penalty_applied.append(penalty.mean().item())
            # add together to get actor loss
            actor_loss = (ent_coef * log_prob - min_qf_pi + penalty).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Train lambda
            if self.dual_interval==self.lam_update_interval:
                self.dual_interval = 0
                with th.no_grad():
                    current_q_values_cost = th.cat(self.critic_cost(replay_data.observations, replay_data.actions), dim=1)
                    violation = th.min(current_q_values_cost, dim=1, keepdim=True)[0] - self.cost_lim
                    # violation = replay_data.costs
                log_lam = F.softplus(self.lam)
                lam_loss = log_lam * violation.detach()
                lam_loss = -lam_loss.mean()
                # lam_loss = lam_loss.mean()

                self.lam_optim.zero_grad()
                lam_loss.backward()
                self.lam_optim.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.critic_cost.parameters(), self.critic_cost_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/lag_lambda", self.lam.item())
        self.logger.record("train/soft_lag_lambda", F.softplus(self.lam).item())
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/critic_cost_loss", np.mean(critic_cost_losses))
        self.logger.record("train/critic_cost", np.mean(critic_costs))
        self.logger.record("train/rce_loss", np.mean(rce_critic_losses))
        self.logger.record("train/rce_pred_pos", np.mean(rce_pred_val))
        self.logger.record("train/rce_pred_neg_pos", np.mean(rce_pred_neg_val))
        self.logger.record("train/rce_rewards_store", np.mean(rce_rewards_store))
        self.logger.record("train/env_rewards_store", np.mean(env_rewards_store))
        self.logger.record("train/penalty_applied", np.mean(penalty_applied))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + \
            ["critic_cost", "critic_cost_target"]
    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = \
            ["policy", "critic_rce_1", "critic_rce_2", "target_critic_rce_1", "target_critic_rce_2", "actor.optimizer", "critic.optimizer", "critic_cost.optimizer", "critic_rce_1.d_optim", "critic_rce_2.d_optim"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables