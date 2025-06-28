from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

from environment.fetch_custom.get_fetch_env import get_env, get_pickplace_env
from environment.general_env import GeneralEnv

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.policies import BasePolicy
from policy.SAC_Lg_Gail.sac_lagrangian_policy import MlpPolicy, CNNPolicy, MultiInputPolicy, SACLagPolicy, Discriminator, ExpertData

import math
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F

import gymnasium as gym

import pdb

class SACLagD(SAC):
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
        d_kwargs = None,
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

        self.d_kwargs = d_kwargs
        
        self.dual_interval = 0
        self.cost_lim = -5e-4

        # only for debug
        # self.observation_space = gym.spaces.Box(-math.inf, math.inf, shape=[16])
        # self.action_space = gym.spaces.Box(-math.inf, math.inf, shape=[4])
        # self.n_envs = 8
        # self.env = env = GeneralEnv(get_pickplace_env(100, eval=False), env_success_rew=0, goal_key='goal')

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

    def _setup_d(self) -> None:
        # init
        # self.d_w = 0.1
        self.d_w = 1.0
        # self.d_w = 100.
        self.d_lr = 0.01
        self.d_net_arch = [100, 100]
        self.d_epoch = 2
        self.d_batch = 1024
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.d_only_obs = True
        self.d_dist = False
        # load
        if self.d_kwargs is not None:
            if 'd_w' in self.d_kwargs:
                self.d_w = self.d_kwargs['d_w']
            if 'd_lr' in self.d_kwargs:
                self.d_lr = self.d_kwargs['d_lr']
            if 'd_net_arch' in self.d_kwargs:
                self.d_net_arch = self.d_kwargs['d_net_arch']
            if 'd_epoch' in self.d_kwargs:
                self.d_epoch = self.d_kwargs['d_epoch']
            if 'd_batch' in self.d_kwargs:
                self.d_batch = self.d_kwargs['d_batch']
            if 'd_only_obs' in self.d_kwargs:
                self.d_only_obs = self.d_kwargs['d_only_obs']
            if 'd_dist' in self.d_kwargs:
                self.d_dist = self.d_kwargs['d_dist']

    def _setup_model(self) -> None:
        # setup policies
        super()._setup_model()

        # setup lambda
        self._setup_lam()
        # setup discriminator
        self._setup_d()

        # setup discriminator
        self.d = Discriminator(self.observation_space, self.action_space, self.d_net_arch, only_obs=self.d_only_obs, \
                               features_extractor=self.actor.features_extractor, optimizer_kwargs={'lr': self.d_lr})
        self.d.to(self.device)

        # add expert data
        if self.replay_buffer is not None and self.ori_exp_data is not None:
            self.replay_buffer.add_exp_data(self.ori_exp_data)
            self.replay_buffer.consider_exp_add(self.exp_data)

        # create aliases
        self._create_aliases()

    def _create_aliases(self) -> None:
        super()._create_aliases()
        self.critic_cost = self.policy.critic_cost
        self.critic_cost_target = self.policy.critic_cost_target

    # rewrite to include lagrangian training (TODO)
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.critic_cost.optimizer, self.d.d_optim]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, critic_cost_losses, lam_loss, d_loss = [], [], [], [], []
        d_pred_val = []
        d_pred_neg_val = []
        d_rewards_store = []
        env_rewards_store = []
        critic_costs = []
        penalty_applied = []
        debug_record = []

        # train discriminator
        for gradient_step in range(self.d_epoch):
            # Sample expert data
            if self.d_dist:
                expert_states, expert_actions, expert_dist = self.exp_data.sample(self.d_batch, only_end=False, with_dist=True)
            else:
                expert_states, expert_actions = self.exp_data.sample(self.d_batch, only_end=False)
                # expert_states, expert_actions = self.exp_data.sample(self.d_batch, only_end=True)
            # Sample replay buffer
            # replay_data = self.replay_buffer.sample(self.d_batch, env=self._vec_normalize_env)
            # traj_states, traj_actions, traj_dones = replay_data.observations, replay_data.actions, replay_data.dones
            replay_data = self.replay_buffer.sample_neg(self.d_batch, env=self._vec_normalize_env)
            traj_states, traj_actions, traj_dones = replay_data.observations, replay_data.actions, replay_data.dones

            # only for debug
            traj_debug = th.logical_and(traj_states[:, -1] > traj_states[:, 21], traj_states[:, -1] < traj_states[:, 24])
            debug_record.append(th.sum(traj_debug).item() / float(traj_debug.shape[0]))

            # pass d and train
            if self.d_dist:
                expert_prob = self.d(expert_states, expert_actions, no_last_act=True)
                # expert_label = th.pow(0.95, expert_dist)
                expert_label = expert_dist
            else:
                expert_prob = self.d(expert_states, expert_actions)
                expert_label = th.ones((expert_prob.shape[0], 1), device=self.device)
                # expert_label = th.zeros((expert_prob.shape[0], 1), device=self.device)

            if self.d_dist:
                expert_loss = self.mse_loss(expert_prob.squeeze(1), expert_label)
            else:
                expert_loss = self.bce_loss(expert_prob, expert_label)

            # traj_label = traj_dones * (th.zeros((traj_prob.shape[0], 1), device=self.device) + 0.5)
            # traj_label = th.zeros((traj_prob.shape[0], 1), device=self.device)
            # traj_label = th.ones((traj_prob.shape[0], 1), device=self.device)
            if self.d_dist:
                traj_prob = self.d(traj_states, traj_actions, no_last_act=True)
                traj_label = th.zeros((traj_prob.shape[0]), device=self.device)
                traj_loss = self.mse_loss(traj_prob.squeeze(1), traj_label)
            else:
                # only for debug
                # pick_ids = (replay_data.rewards <= 0).squeeze(1)
                # traj_prob = self.d(traj_states[pick_ids], traj_actions[pick_ids])
                traj_prob = self.d(traj_states, traj_actions)
                # traj_label = th.ones((traj_prob.shape[0], 1), device=self.device)
                traj_label = th.zeros((traj_prob.shape[0], 1), device=self.device)
                traj_loss = self.bce_loss(traj_prob, traj_label)

            # backward
            loss = expert_loss + traj_loss
            self.d.d_optim.zero_grad()
            loss.backward()
            self.d.d_optim.step()

            # store
            d_loss.append(loss.item())
            d_pred_val.append(expert_prob.squeeze(1).mean().detach().item())
            d_pred_neg_val.append(traj_prob.squeeze(1).mean().detach().item())

        # train policy
        for gradient_step in range(gradient_steps):
            self.dual_interval += 1

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            critic_costs.append(replay_data.costs.mean().item())

            # Get discriminator reward
            with th.no_grad():
                if self.d_dist:
                    d_res = self.d(replay_data.observations, replay_data.actions, no_last_act=True)
                    d_res = th.clip(d_res, min=0., max=1.)
                    next_d_res = self.d(replay_data.next_observations, None, no_last_act=True)
                    next_d_res = th.clip(next_d_res, min=0., max=1.)
                    d_rewards = next_d_res - d_res
                    # d_rewards = d_res
                else:
                    # d_rewards = th.log(self.d(replay_data.observations, replay_data.actions))
                    # d_rewards = th.clip(d_rewards, min=-10., max=0.)
                    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                    d_rewards = self.d(replay_data.next_observations, next_actions)
                    d_rewards = d_rewards - 1
                d_rewards_store.append(d_rewards.mean().item())
                env_rewards_store.append(replay_data.rewards.mean().item())

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

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)

                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + d_rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                # target_q_values = d_rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                # target_q_values = self.d_w * d_rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                # target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                # target_q_values = replay_data.rewards - self.d_w * d_rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                if not self.lam_disable:
                    # Compute the next constrain values: min over all constrains targets
                    next_q_values_cost = th.cat(self.critic_cost_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values_cost, _ = th.min(next_q_values_cost, dim=1, keepdim=True)
                    # add entropy term
                    # next_q_values_cost = next_q_values_cost - ent_coef * next_log_prob.reshape(-1, 1)
                    next_q_values_cost = next_q_values_cost
                    # td erro + entropy term
                    target_q_values_cost = replay_data.costs + (1 - replay_data.dones) * self.gamma * next_q_values_cost
                    # target_q_values_cost = replay_data.costs  + (- self.d_w * d_rewards - 0.5) + (1 - replay_data.dones) * self.gamma * next_q_values_cost
                    # target_q_values_cost = -replay_data.rewards + replay_data.costs + (1 - replay_data.dones) * self.gamma * next_q_values_cost

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
                penalty_applied.append(penalty.mean().detach().item())

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
        self.logger.record("train/d_loss", np.mean(d_loss))
        self.logger.record("train/d_pred_pos", np.mean(d_pred_val))
        self.logger.record("train/d_pred_neg_pos", np.mean(d_pred_neg_val))
        self.logger.record("train/d_rewards_store", np.mean(d_rewards_store))
        self.logger.record("train/env_rewards_store", np.mean(env_rewards_store))
        self.logger.record("train/penalty_applied", np.mean(penalty_applied))
        self.logger.record("train/debug_record", np.mean(debug_record))
        self.logger.record("train/expert_buffer_size", self.exp_data.get_len())
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["critic_cost", "critic_cost_target"]
    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "d", "actor.optimizer", "critic.optimizer", "critic_cost.optimizer", "d.d_optim"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables