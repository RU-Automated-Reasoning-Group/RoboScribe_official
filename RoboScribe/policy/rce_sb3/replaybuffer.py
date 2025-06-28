from typing import NamedTuple, Optional, Union, List, Dict, Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

import pdb

class ExpertReplayBufferLoad:
    def __init__(self, expert_obs, example_num=100):
        self.expert_obs = expert_obs
        self.index = np.arange(example_num)

    def sample(self, batch_size):
        temp_ind = np.random.choice(self.index, batch_size, replace=False)
        batch_obs = self.expert_obs[temp_ind]
        return batch_obs

    @property
    def buffer_size(self):
        return self.expert_obs[0]

class UniformReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    future_observations: th.Tensor


class UniformReplayBuffer(ReplayBuffer):

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(self,         
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = "auto",
                 n_envs: int = 1,
                 optimize_memory_usage: bool = False,
                 handle_timeout_termination: bool = True,):
        super().__init__(buffer_size=buffer_size,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage,
                         handle_timeout_termination=handle_timeout_termination)

    def sample(self, n_steps: int, batch_size: int, env: Optional[VecNormalize] = None) -> UniformReplayBufferSamples:
        assert not self.optimize_memory_usage
        
        upper_bound = self.buffer_size if self.full else self.pos
        done_inds = self.dones[:upper_bound].copy()
        for idx, env_idx in zip(*np.where(done_inds)):
            if idx < n_steps-1:
                done_inds[:idx, env_idx] = True
                if self.full:
                    done_inds[-(n_steps-1-idx):, env_idx]
            else:
                done_inds[idx-(n_steps-1): idx, env_idx] = True

        valid_inds = np.where(1 - done_inds)
        valid_inds_idxs = np.arange(valid_inds[0].shape[0])
        pick_idxs = np.random.choice(valid_inds_idxs, size=batch_size)
        batch_inds, env_inds = valid_inds[0][pick_idxs], valid_inds[1][pick_idxs]

        # upper_bound = self.buffer_size if self.full else self.pos-n_steps+1
        # batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        # return self._get_samples(n_steps, batch_inds, env=env)

        return self._get_samples(n_steps, batch_inds, env_inds, env=env)


    def _get_samples(self, n_steps: int, batch_inds: np.ndarray, env_inds: np.ndarray, env: Optional[VecNormalize] = None) -> UniformReplayBufferSamples:
        assert not self.optimize_memory_usage

        # Next States
        # next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_inds, :], env)

        # Future States
        future_idxs = (batch_inds + n_steps - 1) % self.buffer_size
        future_obs = self._normalize_obs(self.next_observations[future_idxs, env_inds, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_inds, :], env),
            self.actions[batch_inds, env_inds, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_inds] * (1 - self.timeouts[batch_inds, env_inds])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_inds].reshape(-1, 1), env),
            future_obs
        )
        return UniformReplayBufferSamples(*tuple(map(self.to_torch, data)))