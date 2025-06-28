from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.noise import ActionNoise
from policy.SAC_Bag_new.sac_lagrangian_policy import MlpPolicy, CNNPolicy, MultiInputPolicy, SACPolicyBag
from stable_baselines3.sac.policies import SACPolicy
from policy.commons.buffer import ReplayBufferLagBag, ReplayBufferSimpleBag
from gymnasium import spaces

import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F

import pdb

class SACBag(SAC):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": SACPolicy,
        "MultiInputPolicy": SACPolicy,
    }
    policy: SACPolicy

    def __init__(
        self,
        policy_num = 2,
        num_envs = 1,
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

        # only support n_env == 1
        assert self.n_envs == 1
        self.policy_leverage = {env_id: 0 for env_id in range(num_envs)}

        # setup model
        self.ori_exp_data = exp_data
        if _init_setup_model:
            self._setup_model()

    # setup model (done)
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # policy bag
        policy_dict = {}
        replay_buffer_dict = {}
        entropy_dict = {}
        for policy_id in range(self.policy_num):
            # replay buffer
            if self.replay_buffer_class is None:
                if isinstance(self.observation_space, spaces.Dict):
                    self.replay_buffer_class = DictReplayBuffer
                else:
                    self.replay_buffer_class = ReplayBuffer
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            if issubclass(self.replay_buffer_class, HerReplayBuffer):
                assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
                replay_buffer_kwargs["env"] = self.env
            replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,
            )

            # policy
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

            # store
            replay_buffer_dict[str(policy_id)] = replay_buffer
            policy_dict[str(policy_id)] = new_policy

            cur_entropy_dict['batch_norm_stats'] = new_batch_norm_stats
            cur_entropy_dict['batch_norm_stats_target'] = new_batch_norm_stats_target
            cur_entropy_dict['target_entropy'] = new_target_entropy
            entropy_dict[policy_id] = cur_entropy_dict

        self.policy = SACPolicyBag(nn.ModuleDict(policy_dict), self.policy_leverage)
        self.replay_buffer = ReplayBufferSimpleBag(self.buffer_size,
                                                   self.observation_space,
                                                   self.action_space,
                                                   device=self.device,
                                                   n_envs=self.n_envs,
                                                   optimize_memory_usage=self.optimize_memory_usage,
                                                   **replay_buffer_kwargs,
                                                   replay_buffer_dict=replay_buffer_dict)
        self.entropy_dict = entropy_dict

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

    # load policies (done)
    def load_policy(self, policy_dict):
        self.policy.load_policy(policy_dict)

    # store all ids (done)
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

    # train specific policy
    def train_policy(self, gradient_steps, batch_size, policy_id):
        # get policy of related id
        policy = self.policy.policy_dict[str(policy_id)]
        actor = policy.actor
        critic = policy.critic

        # init policy
        policy = self.policy.policy_dict[str(policy_id)]
        actor = policy.actor
        critic = policy.critic
        critic_target = policy.critic_target
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

        # Switch to train mode (this affects batch norm / dropout)
        policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [actor.optimizer, critic.optimizer]
        if ent_coef_optimizer is not None:
            optimizers += [ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, policy_id=policy_id, env=self._vec_normalize_env)  # type: ignore[union-attr]

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

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            actor.optimizer.zero_grad()
            actor_loss.backward()
            actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(critic.parameters(), critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(batch_norm_stats, batch_norm_stats_target, 1.0)

        self.logger.record(f"train/ent_coef_{policy_id}", np.mean(ent_coefs))
        self.logger.record(f"train/actor_loss_{policy_id}", np.mean(actor_losses))
        self.logger.record(f"train/critic_loss_{policy_id}", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record(f"train/ent_coef_loss_{policy_id}", np.mean(ent_coef_losses))

    # train all policies
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        for policy_id in range(self.policy_num):
            if self.replay_buffer.check_available(policy_id, batch_size):
                self.train_policy(gradient_steps, batch_size, policy_id)
            self.logger.record(f"train/replay_buffer_{policy_id}_size", self.replay_buffer.replay_buffer_dict[str(policy_id)].size())

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    # add demo data into replay buffer
    def add_transitions(self, datasets, get_info=False):
        # support num_env == 1 for now
        assert len(self.policy_leverage) == 1
        for policy_id, dataset in enumerate(datasets):
            # init
            observations = dataset['observations']
            actions = dataset['actions']
            rewards = dataset['rewards']
            next_observations = dataset['next_observations']
            dones = dataset['terminals']
            # store
            for obs, next_obs, action, reward, done in zip(observations, next_observations, actions, rewards, dones):
                self.replay_buffer.replay_buffer_dict[str(policy_id)].add(np.expand_dims(obs, 0),
                                                                          np.expand_dims(next_obs, 0), 
                                                                          np.expand_dims(action, 0), 
                                                                          np.expand_dims(reward, 0), 
                                                                          np.expand_dims(done, 0), 
                                                                          [{}])
            # print information
            if get_info:
                print('replay buffer size for {} is: {}'.format(policy_id, self.replay_buffer.replay_buffer_dict[str(policy_id)].size()))

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