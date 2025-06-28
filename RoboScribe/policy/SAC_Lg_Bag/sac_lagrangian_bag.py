from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.noise import ActionNoise
from policy.SAC_Lg_Bag.sac_lagrangian_policy import MlpPolicy, CNNPolicy, MultiInputPolicy, SACLagPolicy, SACLagPolicyBag
from policy.commons.buffer import ReplayBufferLagBag
from gymnasium import spaces

import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F

import pdb

class SACLagBag(SAC):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": SACLagPolicy,
        "MultiInputPolicy": SACLagPolicy,
    }
    policy: SACLagPolicy

    def __init__(
        self,
        policy_num = 2,
        num_envs = 8,
        lam_disable = False,
        lam_init = 10.0,
        lam_update_interval = 12,
        lam_lr = 0.03,
        exp_data = None,
        _init_setup_model = True,
        *args,
        **kwargs,
    ):
        # init for SAC
        if '_init_setup_model' in kwargs:
            del kwargs['_init_setup_model']
        super().__init__(_init_setup_model=False, *args, **kwargs)

        # init for bag
        self.policy_num = policy_num
        self.policy_leverage = {env_id: 0 for env_id in range(self.env.num_envs)}
        # self.policy_leverage = {env_id: 0 for env_id in range(num_envs)}

        # init for lagrangian
        self.lam_disable = lam_disable
        self.lam_init = lam_init
        self.lam_update_interval = lam_update_interval
        self.lam_lr = lam_lr
        
        self.dual_interval = 0
        self.cost_lim = -5e-4

        # setup model
        self.ori_exp_data = exp_data
        if _init_setup_model:
            self._setup_model()

    # setup model
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

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
        else:
            assert isinstance(self.replay_buffer, ReplayBufferLagBag)

        # add expert data
        if self.replay_buffer is not None and self.ori_exp_data is not None:
            self.replay_buffer.add_exp_data(self.ori_exp_data)

        # policy bag
        policy_dict = {}
        entropy_dict = {}
        lam_dict = {}
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
                new_target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
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
            policy_dict[str(policy_id)] = new_policy

            cur_entropy_dict['batch_norm_stats'] = new_batch_norm_stats
            cur_entropy_dict['batch_norm_stats_target'] = new_batch_norm_stats_target
            cur_entropy_dict['target_entropy'] = new_target_entropy
            entropy_dict[policy_id] = cur_entropy_dict

            cur_lam_dict = {'lam': new_lam, 'lam_optim': new_lam_optim}
            lam_dict[policy_id] = cur_lam_dict

        policy_dict = nn.ModuleDict(policy_dict)
        self.policy = SACLagPolicyBag(policy_dict, env_policy_used=self.policy_leverage)
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
        replay_buffer: ReplayBuffer,
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

    # rewrite to include lagrangian training
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # train for each policy
        self.dual_interval += gradient_steps
        for policy_id in range(self.policy_num):
            # check whether policy valid
            if not self.replay_buffer.check_available(policy_id, batch_size):
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

            # Switch to train mode (this affects batch norm / dropout)
            policy.set_training_mode(True)
            # Update optimizers learning rate
            optimizers = [actor.optimizer, critic.optimizer, critic_cost.optimizer]
            if ent_coef_optimizer is not None:
                optimizers += [ent_coef_optimizer]

            # Update learning rate according to lr schedule
            self._update_learning_rate(optimizers)

            ent_coef_losses, ent_coefs = [], []
            actor_losses, critic_losses, critic_cost_losses, lam_loss = [], [], [], []
            critic_costs = []
            penalty_applied = []

            for gradient_step in range(gradient_steps):
                # Sample replay buffer
                replay_data = self.replay_buffer.sample(batch_size, policy_id=policy_id, env=self._vec_normalize_env)  # type: ignore[union-attr]
                critic_costs.append(replay_data.costs.mean().item())

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
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                    if not self.lam_disable:
                        # Compute the next constrain values: min over all constrains targets
                        next_q_values_cost = th.cat(critic_cost_target(replay_data.next_observations, next_actions), dim=1)
                        next_q_values_cost, _ = th.min(next_q_values_cost, dim=1, keepdim=True)
                        # add entropy term
                        # next_q_values_cost = next_q_values_cost - ent_coef * next_log_prob.reshape(-1, 1)
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

                if not self.lam_disable:
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
                if self.lam_disable:
                    penalty = 0
                    penalty_applied.append(penalty)
                else:
                    # calculate penalty from cost
                    q_values_cost_pi = th.cat(critic_cost(replay_data.observations, actions_pi), dim=1)
                    min_qf_cost_pi, _ = th.min(q_values_cost_pi, dim=1, keepdim=True)
                    # penalty = max(self.lam.item(), 0) * min_qf_cost_pi
                    penalty = F.softplus(lam).item() * min_qf_cost_pi
                    penalty_applied.append(penalty.mean().detach().item())

                # add together to get actor loss
                actor_loss = (ent_coef * log_prob - min_qf_pi + penalty).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                actor.optimizer.zero_grad()
                actor_loss.backward()
                actor.optimizer.step()

                # Train lambda
                if self.dual_interval >= self.lam_update_interval:
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
            self.logger.record("train/penalty_applied_{}".format(policy_id), np.mean(penalty_applied))
            if len(ent_coef_losses) > 0:
                self.logger.record("train/ent_coef_loss_{}".format(policy_id), np.mean(ent_coef_losses))

        # update
        if self.dual_interval >= self.lam_update_interval:
            self.dual_interval = 0

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

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
        state_dicts = ["policy"]
        return state_dicts, []