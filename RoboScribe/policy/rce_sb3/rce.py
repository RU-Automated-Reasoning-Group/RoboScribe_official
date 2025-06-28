from typing import Any, ClassVar, Dict, List, Type

import torch as th
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import polyak_update

from policy.rce_sb3.rce_policy import MlpPolicy, CNNPolicy, MultiInputPolicy, RCEPolicy

import pdb

class RCE(SAC):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": RCEPolicy,
        "MultiInputPolicy": RCEPolicy,
    }
    policy: RCEPolicy

    def __init__(self, 
                 expert_examples_buffer=None,
                 future_step=None,
                 *args,
                 **kwargs):
        # init for SAC
        super().__init__(*args, **kwargs)

        # init for RCE
        self.expert_buffer = expert_examples_buffer
        self.future_step = future_step
        self.critic_criterion = th.nn.MSELoss(reduction='none')

    # rewrite to train
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        pdb.set_trace()

        for gradient_step in range(gradient_steps):
            # sample expert examples
            expert_states = th.tensor(self.expert_buffer.sample(batch_size)).to(self.device)

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(self.future_step, batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

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
                next_actions, _ = self.actor.action_log_prob(replay_data.next_observations)
                # compute the next Q values
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)

                # compute the future q values
                future_actions, _ = self.actor.action_log_prob(replay_data.future_observations)
                future_q_values = th.cat(self.critic_target(replay_data.future_observations, future_actions), dim=1)

                # compute target q
                gamma_n = self.gamma * self.future_step
                target_q_values = (next_q_values + gamma_n * future_q_values) / 2.0
                target_q_values, _ = th.min(target_q_values, dim=1, keepdim=True)

                # compute td prob
                w = target_q_values / (1. - target_q_values)
                td_target_values = self.gamma * w / (1. + self.gamma * w)

            td_target_values = th.cat([th.ones(batch_size, 1).to(self.device), td_target_values], dim=0)
            weights = th.cat(
                [th.ones(batch_size, 1).to(self.device) - self.gamma, 1. + self.gamma * w], dim=0)

            # Compute predictions of expert states
            expert_actions, _ = self.actor.action_log_prob(expert_states)
            expert_q_values = self.critic(expert_states, expert_actions)

            # Computer current q values
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # compute critic loss
            current_q_values_1 = th.cat([expert_q_values[0], current_q_values[0]], dim=0)
            critic_loss_1 = (weights * self.critic_criterion(current_q_values_1, td_target_values)).mean()
            current_q_values_2 = th.cat([expert_q_values[1], current_q_values[1]], dim=0)
            critic_loss_2 = (weights * self.critic_criterion(current_q_values_2, td_target_values)).mean()

            # optimize critic network
            critic_loss = (critic_loss_1 + critic_loss_2) / 2
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)
