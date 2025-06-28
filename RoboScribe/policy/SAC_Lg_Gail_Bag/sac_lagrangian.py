from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

from environment.fetch_custom.get_fetch_env import get_env, get_pickplace_env
from environment.general_env import GeneralEnv

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name
from stable_baselines3.common.policies import BasePolicy
from policy.commons.buffer import ReplayBufferLagBag, ReplayBufferLagBagDebug
from policy.SAC_Lg_Gail_Bag.sac_lagrangian_policy import MlpPolicy, CNNPolicy, MultiInputPolicy, SACLagPolicy, Discriminator, ExpertData, SACLagPolicyBag, DiscriminatorBag

import copy
import math
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt

import gymnasium as gym
import os

import pdb

class SACLagBagD(SAC):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": SACLagPolicy,
        "MultiInputPolicy": SACLagPolicy,
    }
    policy: SACLagPolicy

    def __init__(
        self,
        policy_num = 2,
        lam_disable = False,
        exp_data_dict = None,
        d_kwargs = None,
        lam_kwargs = None,
        _init_setup_model = True,
        behavior_clone = False,
        *args,
        **kwargs,
    ):
        # init for SAC
        if '_init_setup_model' in kwargs:
            del kwargs['_init_setup_model']
        super().__init__(
            _init_setup_model=False, *args, **kwargs
        )

        # init for bag
        self.policy_num = policy_num

        # init for lagrangian
        self.lam_kwargs = lam_kwargs
        self.lam_disable = lam_disable

        # init for discriminator
        self.ori_exp_data_dict = exp_data_dict
        if self.ori_exp_data_dict is not None:
            self.exp_data_dict = {data_id:ExpertData(self.ori_exp_data_dict[data_id], self.device) \
                                  for data_id in self.ori_exp_data_dict}
            self.copy_exp_data_dict = copy.deepcopy(self.exp_data_dict)

        self.d_kwargs = d_kwargs
        
        self.dual_interval = 0
        self.cost_lim = -5e-4

        # setup model
        if _init_setup_model:
            self._setup_model()

        # only for debug
        self.behavior_clone = behavior_clone
        # plt.figure()
        # self.debug_id = 0
        # self.debug_store_path = 'store/debug_0'
        # if not os.path.exists(self.debug_store_path):
        #     os.makedirs(self.debug_store_path)

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
        # self.lam = th.tensor(self.lam_init, requires_grad=True)
        # self.lam_optim = th.optim.Adam([self.lam], lr=self.lam_lr)

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
        # self.d_only_obs = False
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

    # setup model
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # init for bag
        self.policy_leverage = {env_id: 0 for env_id in range(self.n_envs)}

        # set up parameters
        self._setup_d()
        self._setup_lam()

        # replay buffer
        self.replay_buffer_class = ReplayBufferLagBag
        if self.replay_buffer is None:
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                policy_num=self.policy_num,
                **replay_buffer_kwargs,
            )
            if self.ori_exp_data_dict is not None:
                self.replay_buffer.consider_exp_add(self.exp_data_dict)
            # only for debug
            # self.replay_buffer = self.replay_buffer_class(
            #     self.buffer_size,
            #     self.observation_space,
            #     self.action_space,
            #     device=self.device,
            #     n_envs=self.n_envs,
            #     optimize_memory_usage=self.optimize_memory_usage,
            #     **replay_buffer_kwargs,
            # )
            # self.replay_buffer.consider_exp_add(self.exp_data_dict[0])
        else:
            assert isinstance(self.replay_buffer, ReplayBufferLagBag)

        # add expert data
        if self.replay_buffer is not None and self.ori_exp_data_dict is not None:
            self.replay_buffer.add_exp_data(self.ori_exp_data_dict)
            # only for debug
            # self.replay_buffer.add_exp_data(self.ori_exp_data_dict[0])

        # policy bag
        policy_dict = {}
        entropy_dict = {}
        lam_dict = {}
        d_dict = {}
        for policy_id in range(self.policy_num):
            new_policy = self.policy_class(
                                        self.observation_space,
                                        self.action_space,
                                        self.lr_schedule,
                                        **self.policy_kwargs,
                                    )
            new_policy.to(self.device)

            # Running mean and running var
            new_batch_norm_stats = get_parameters_by_name(new_policy.critic, ["running_"])
            new_batch_norm_stats_target = get_parameters_by_name(new_policy.critic_target, ["running_"])
            # Target entropy is used when learning the entropy coefficient
            if self.target_entropy == "auto":
                # automatically set target entropy if needed
                new_target_entropy = float(-np.prod(self.action_space.shape).astype(np.float32))  # type: ignore
            else:
                # Force conversion
                # this will also throw an error for unexpected string
                new_target_entropy = float(self.target_entropy)

            # The entropy coefficient or entropy can be learned automatically
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
                # Default initial value of ent_coef when learned
                init_value = 1.0
                if "_" in self.ent_coef:
                    init_value = float(self.ent_coef.split("_")[1])
                    assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

                # Note: we optimize the log of the entropy coeff which is slightly different from the paper
                # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                new_log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
                new_ent_coef_optimizer = th.optim.Adam([new_log_ent_coef], lr=self.lr_schedule(1))
                # store
                cur_entropy_dict = {'log_ent_coef': new_log_ent_coef, 'ent_coef_optimizer': new_ent_coef_optimizer}
            else:
                # Force conversion to float
                # this will throw an error if a malformed string (different from 'auto')
                # is passed
                new_ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)
                # store
                cur_entropy_dict = {'ent_coef_tensor': new_ent_coef_tensor}

            # lambda store
            new_lam = th.tensor(self.lam_init, requires_grad=True)
            new_lam_optim = th.optim.Adam([new_lam], lr=self.lam_lr)

            # store
            # policy
            policy_dict[str(policy_id)] = new_policy
            # entropy
            cur_entropy_dict['batch_norm_stats'] = new_batch_norm_stats
            cur_entropy_dict['batch_norm_stats_target'] = new_batch_norm_stats_target
            cur_entropy_dict['target_entropy'] = new_target_entropy
            entropy_dict[policy_id] = cur_entropy_dict
            # lambda
            cur_lam_dict = {'lam': new_lam, 'lam_optim': new_lam_optim}
            lam_dict[policy_id] = cur_lam_dict
            # discriminator
            new_d = Discriminator(self.observation_space, self.action_space, self.d_net_arch, only_obs=self.d_only_obs, \
                               features_extractor=new_policy.actor.features_extractor, optimizer_kwargs={'lr': self.d_lr})
            new_d.to(self.device)
            d_dict[str(policy_id)] = new_d

        policy_dict = nn.ModuleDict(policy_dict)
        self.policy = SACLagPolicyBag(policy_dict, env_policy_used=self.policy_leverage)
        d_dict = nn.ModuleDict(d_dict)
        self.d = DiscriminatorBag(d_dict)
        self.entropy_dict = entropy_dict
        self.lam_dict = lam_dict

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

    # load policies
    def load_policy(self, policy_dict):
        self.policy.load_policy(policy_dict)

    # store all ids
    def _store_transition(
        self,
        replay_buffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        for env_id in self.policy_leverage:
            self.policy_leverage[env_id] = infos[env_id]['policy_id']
        self.policy.env_policy_used = self.policy_leverage

        # for env_id in range(len(infos)):
        #     if infos[env_id]['state_store'] == 'store':
        #         store_img = self.env.envs[env_id].render()
        #         plt.imshow(store_img)
        #         plt.savefig(os.path.join(self.debug_store_path, '{}.png'.format(self.debug_id)))
        #         plt.cla()
        #         self.debug_id += 1

        # store buffer information for each policy
        policy_num_dict = self.replay_buffer.get_spec_num()
        stage_rewards = self.replay_buffer.get_stage_reward()
        for policy_id in policy_num_dict:
            self.logger.record("rollout/buffer_policy_num_{}".format(policy_id), np.mean(policy_num_dict[policy_id]))
            self.logger.record("rollout/stage_reward_{}".format(policy_id), stage_rewards[policy_id])

    # evaluate log-prob and entropy of given observation and action
    def evaluate_actions(self, actor, obs, actions):
        # process observation
        mean_actions, log_std, kwargs = actor.get_action_dist_params(obs)
        # create distribution
        actor.action_dist.proba_distribution(mean_actions, log_std)
        # log_prob = actor.action_dist.log_prob(actions)
        log_prob = actor.action_dist.distribution.log_prob(actions).sum(dim=1)
        log_prob[log_prob!=log_prob] = 0
        # entropy = actor.action_dist.entropy()
        entropy = actor.action_dist.distribution.entropy().sum(dim=1)

        return log_prob, entropy

    # rewrite to include lagrangian training (TODO)
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # train for each policy
        for policy_id in range(self.policy_num):
            # check whether policy valid
            if not self.replay_buffer.check_available(policy_id, self.learning_starts):
                continue

            # init policy
            policy = self.policy.policy_dict[str(policy_id)]
            actor = policy.actor
            critic = policy.critic
            critic_target = policy.critic_target
            critic_cost = policy.critic_cost
            critic_cost_target = policy.critic_cost_target
            # init entropy
            entropy_dict = self.entropy_dict[policy_id]
            batch_norm_stats = entropy_dict['batch_norm_stats']
            batch_norm_stats_target = entropy_dict['batch_norm_stats_target']
            target_entropy = entropy_dict['target_entropy']
            if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
                log_ent_coef = entropy_dict['log_ent_coef']
                ent_coef_optimizer = entropy_dict['ent_coef_optimizer']
            else:
                ent_coef_optimizer = None
                log_ent_coef = None
                ent_coef_tensor = entropy_dict['ent_coef_tensor']
            # init lambda
            lam_dict = self.lam_dict[policy_id]
            lam = lam_dict['lam']
            lam_optim = lam_dict['lam_optim']
            # init discriminator
            d_net = self.d.d_dict[str(policy_id)]

            # Switch to train mode (this affects batch norm / dropout)
            policy.set_training_mode(True)
            # Update optimizers learning rate
            optimizers = [actor.optimizer, critic.optimizer, critic_cost.optimizer, d_net.d_optim]
            if ent_coef_optimizer is not None:
                optimizers += [ent_coef_optimizer]

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
            behavior_clone_losses = []
            # debug_record = []

            # train discriminator
            for gradient_step in range(self.d_epoch):
                # Sample expert data
                expert_states, expert_actions = self.exp_data_dict[policy_id].sample(self.d_batch, only_end=False)

                # Sample replay buffer
                # replay_data = self.replay_buffer.sample(self.d_batch, policy_id, env=self._vec_normalize_env)
                neg_replay_data = self.replay_buffer.sample_neg(self.d_batch, policy_id, env=self._vec_normalize_env)
                # only for debug
                # neg_replay_data = self.replay_buffer.sample_neg(self.d_batch, env=self._vec_normalize_env)
                traj_states, traj_actions, traj_dones = neg_replay_data.observations, neg_replay_data.actions, neg_replay_data.dones

                # only for debug
                # traj_debug = th.logical_and(traj_states[:, -1] > traj_states[:, 21], traj_states[:, -1] < traj_states[:, 24])
                # debug_record.append(th.sum(traj_debug).item() / float(traj_debug.shape[0]))

                # pass d and train
                expert_prob = d_net(expert_states, expert_actions)
                expert_label = th.ones((expert_prob.shape[0], 1), device=self.device)

                if self.d_dist:
                    expert_loss = self.mse_loss(expert_prob.squeeze(1), expert_label)
                else:
                    expert_loss = self.bce_loss(expert_prob, expert_label)

                # only for debug
                # pick_ids = (replay_data.rewards <= 0).squeeze(1)
                # traj_prob = d_net(traj_states[pick_ids], traj_actions[pick_ids])
                traj_prob = d_net(traj_states, traj_actions)
                traj_label = th.zeros((traj_prob.shape[0], 1), device=self.device)
                traj_loss = self.bce_loss(traj_prob, traj_label)

                # backward
                loss = expert_loss + traj_loss
                d_net.d_optim.zero_grad()
                loss.backward()
                d_net.d_optim.step()

                # store
                d_loss.append(loss.item())
                d_pred_val.append(expert_prob.squeeze(1).mean().detach().item())
                d_pred_neg_val.append(traj_prob.squeeze(1).mean().detach().item())

            # train policy
            for gradient_step in range(gradient_steps):
                # Sample replay buffer
                replay_data = self.replay_buffer.sample(batch_size, policy_id, env=self._vec_normalize_env)  # type: ignore[union-attr]
                # only for debug
                # replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                critic_costs.append(replay_data.costs.mean().item())

                # Get discriminator reward
                with th.no_grad():
                    if self.d_only_obs:
                        next_actions, next_log_prob = actor.action_log_prob(replay_data.next_observations)
                        d_rewards = d_net(replay_data.next_observations, next_actions)
                    else:
                        d_rewards = d_net(replay_data.observations, replay_data.actions)
                    d_rewards = d_rewards - 1
                    d_rewards_store.append(d_rewards.mean().item())
                    env_rewards_store.append(replay_data.rewards.mean().item())

                # We need to sample because `log_std` may have changed between two gradient steps
                if self.use_sde:
                    actor.reset_noise()

                # Action by the current actor for the sampled state
                actions_pi, log_prob = actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)

                ent_coef_loss = None
                if ent_coef_optimizer is not None and log_ent_coef is not None:
                    # Important: detach the variable from the graph
                    # so we don't change it with other losses
                    # see https://github.com/rail-berkeley/softlearning/issues/60
                    ent_coef = th.exp(log_ent_coef.detach())
                    ent_coef_loss = -(log_ent_coef * (log_prob + target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                else:
                    ent_coef = ent_coef_tensor

                ent_coefs.append(ent_coef.item())

                # Optimize entropy coefficient, also called
                # entropy temperature or alpha in the paper
                if ent_coef_loss is not None and ent_coef_optimizer is not None:
                    ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    ent_coef_optimizer.step()

                with th.no_grad():
                    # Select action according to policy
                    next_actions, next_log_prob = actor.action_log_prob(replay_data.next_observations)

                    # Compute the next Q values: min over all critics targets
                    next_q_values = th.cat(critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    # add entropy term
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    # td error + entropy term
                    target_q_values = self.d_w * replay_data.rewards + d_rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                    # target_q_values = d_rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                    # target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                    if not self.lam_disable[policy_id]:
                        # Compute the next constrain values: min over all constrains targets
                        next_q_values_cost = th.cat(critic_cost_target(replay_data.next_observations, next_actions), dim=1)
                        next_q_values_cost, _ = th.min(next_q_values_cost, dim=1, keepdim=True)
                        # add entropy term
                        next_q_values_cost = next_q_values_cost
                        # td erro + entropy term
                        target_q_values_cost = replay_data.costs + (1 - replay_data.dones) * self.gamma * next_q_values_cost

                # Get current Q-values estimates for each critic network
                # using action from the replay buffer
                current_q_values = critic(replay_data.observations, replay_data.actions)
                # Compute critic loss
                critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
                assert isinstance(critic_loss, th.Tensor)  # for type checker
                critic_losses.append(critic_loss.item())  # type: ignore[union-attr]
                # Optimize the critic
                critic.optimizer.zero_grad()
                critic_loss.backward()
                critic.optimizer.step()

                if not self.lam_disable[policy_id]:
                    # get current cost
                    current_q_values_cost = critic_cost(replay_data.observations, replay_data.actions)
                    # Compute critic cost loss
                    critic_cost_loss = 0.5 * sum(F.mse_loss(current_q_cost, target_q_values_cost) for current_q_cost in current_q_values_cost)
                    assert isinstance(critic_cost_loss, th.Tensor)
                    critic_cost_losses.append(critic_cost_loss.item())
                    # Optimize the critic cost
                    critic_cost.optimizer.zero_grad()
                    critic_cost_loss.backward()
                    critic_cost.optimizer.step()

                # Compute actor loss
                # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
                # Min over all critic networks
                q_values_pi = th.cat(critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                if self.lam_disable[policy_id]:
                    penalty = 0
                    penalty_applied.append(penalty)
                else:
                    # calculate penalty from cost
                    q_values_cost_pi = th.cat(critic_cost(replay_data.observations, actions_pi), dim=1)
                    min_qf_cost_pi, _ = th.min(q_values_cost_pi, dim=1, keepdim=True)
                    penalty = F.softplus(lam).item() * min_qf_cost_pi
                    penalty_applied.append(penalty.mean().detach().item())

                # add together to get actor loss
                actor_loss = (ent_coef * log_prob - min_qf_pi + penalty).mean()
                actor_losses.append(actor_loss.item())

                # behavior clone to test
                if self.behavior_clone:
                    # Sample expert data
                    # expert_states, expert_actions = self.exp_data_dict[policy_id].sample(self.d_batch, only_end=False)
                    expert_states, expert_actions = self.copy_exp_data_dict[policy_id].sample(self.d_batch, only_end=False)
                    # mean_expert_action, _, _ = actor.get_action_dist_params(expert_states)
                    # behavior_clone_loss = F.mse_loss(mean_expert_action, expert_actions)
                    # actor_loss += 10 * behavior_clone_loss

                    expert_log_prob, expert_entropy = self.evaluate_actions(actor, expert_states, expert_actions)
                    l2_norms = [th.sum(th.square(w)) for w in actor.parameters()]
                    l2_norm = sum(l2_norms) / 2
                    # behavior_clone_loss = -expert_log_prob.mean() - expert_entropy.mean() - l2_norm
                    behavior_clone_loss = -expert_log_prob.mean() - 0.1 * expert_entropy.mean() + 1e-4 * l2_norm
                    actor_loss += behavior_clone_loss

                    behavior_clone_losses.append(behavior_clone_loss.item())

                # Optimize the actor
                actor.optimizer.zero_grad()
                actor_loss.backward()
                actor.optimizer.step()

                # Train lambda
                if self.dual_interval>=self.lam_update_interval:
                    self.dual_interval = 0
                    with th.no_grad():
                        current_q_values_cost = th.cat(critic_cost(replay_data.observations, replay_data.actions), dim=1)
                        violation = th.min(current_q_values_cost, dim=1, keepdim=True)[0] - self.cost_lim
                        # violation = replay_data.costs
                    log_lam = F.softplus(lam)
                    lam_loss = log_lam * violation.detach()
                    lam_loss = -lam_loss.mean()
                    # lam_loss = lam_loss.mean()

                    lam_optim.zero_grad()
                    lam_loss.backward()
                    lam_optim.step()

                # Update target networks
                if gradient_step % self.target_update_interval == 0:
                    polyak_update(critic.parameters(), critic_target.parameters(), self.tau)
                    polyak_update(critic_cost.parameters(), critic_cost_target.parameters(), self.tau)
                    # Copy running stats, see GH issue #996
                    polyak_update(batch_norm_stats, batch_norm_stats_target, 1.0)

            self.logger.record("train/ent_coef_{}".format(policy_id), np.mean(ent_coefs))
            self.logger.record("train/lag_lambda_{}".format(policy_id), lam.item())
            self.logger.record("train/soft_lag_lambda_{}".format(policy_id), F.softplus(lam).item())
            self.logger.record("train/actor_loss_{}".format(policy_id), np.mean(actor_losses))
            self.logger.record("train/critic_loss_{}".format(policy_id), np.mean(critic_losses))
            self.logger.record("train/critic_cost_loss_{}".format(policy_id), np.mean(critic_cost_losses))
            self.logger.record("train/critic_cost_{}".format(policy_id), np.mean(critic_costs))
            self.logger.record("train/d_loss_{}".format(policy_id), np.mean(d_loss))
            self.logger.record("train/d_pred_pos_{}".format(policy_id), np.mean(d_pred_val))
            self.logger.record("train/d_pred_neg_pos_{}".format(policy_id), np.mean(d_pred_neg_val))
            self.logger.record("train/d_rewards_store_{}".format(policy_id), np.mean(d_rewards_store))
            self.logger.record("train/env_rewards_store_{}".format(policy_id), np.mean(env_rewards_store))
            self.logger.record("train/penalty_applied_{}".format(policy_id), np.mean(penalty_applied))
            # self.logger.record("train/debug_record_{}".format(policy_id), np.mean(debug_record))
            self.logger.record("train/expert_buffer_size_{}".format(policy_id), self.exp_data_dict[policy_id].get_len())
            if len(ent_coef_losses) > 0:
                self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            if self.behavior_clone:
                self.logger.record("train/behavior_clone_loss_{}".format(policy_id), np.mean(behavior_clone_losses))

        self._n_updates += gradient_steps
        self.dual_interval += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # only for debug
        self.logger.record("rollout/last_pos_comp", self.replay_buffer.pos)
        for env_id in range(len(self.replay_buffer.last_store_pos)):
            self.logger.record("rollout/last_pos_{}".format(env_id), self.replay_buffer.last_store_pos[env_id])

    # pure train to debug
    def pure_clone_train(self):
        all_behavior_clone_losses = []
        for policy_id in range(self.policy_num):
            # init policy
            policy = self.policy.policy_dict[str(policy_id)]
            actor = policy.actor
            # Update learning rate according to lr schedule
            # self._update_learning_rate([actor.optimizer])
            # Sample expert data
            expert_states, expert_actions = self.exp_data_dict[policy_id].sample(self.d_batch, only_end=False)
            # mean_expert_action, _, _ = actor.get_action_dist_params(expert_states)
            # behavior_clone_loss = F.mse_loss(mean_expert_action, expert_actions)
            expert_log_prob, expert_entropy = self.evaluate_actions(actor, expert_states, expert_actions)
            l2_norms = [th.sum(th.square(w)) for w in actor.parameters()]
            l2_norm = sum(l2_norms) / 2
            behavior_clone_loss = - expert_log_prob.mean() - 0.1 * expert_entropy.mean() + 1e-4 * l2_norm
            # behavior_clone_loss = -expert_log_prob.mean()

            # update
            actor.optimizer.zero_grad()
            behavior_clone_loss.backward()
            actor.optimizer.step()
            all_behavior_clone_losses.append(behavior_clone_loss.item())
            # record
            # self.logger.record("train/behavior_clone_loss", behavior_clone_loss.item())

        return all_behavior_clone_losses

    # rewrite predict action
    def predict(self,
                observation: Union[np.ndarray, Dict[str, np.ndarray]],
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False,
                policy_id = None):
        return self.policy.predict(observation, state, episode_start, deterministic, policy_id)

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params()
    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "d"]
        return state_dicts, []