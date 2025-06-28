from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from typing import NamedTuple, List, Dict, Any, Optional, Union
from gymnasium import spaces
import torch as th
import numpy as np

import pdb

class ReplayBufferLagSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    costs: th.Tensor

class ReplayBuffferLag(ReplayBuffer):
    costs: np.ndarray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.costs[self.pos] = np.array([info['cost'] for info in infos])
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
        
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferLagSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self._normalize_reward(self.costs[batch_inds, env_indices].reshape(-1, 1), env)
        )
        return ReplayBufferLagSamples(*tuple(map(self.to_torch, data)))

class ReplayBufferLagD(ReplayBuffer):
    costs: np.ndarray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.exp_data = {'obs': None, 'next_obs': None, 'action': None, 'reward': None, 'cost': None, 'done': None}
        self.exp_len = 0
        self.do_exp_add = False
        self.last_store_pos = np.zeros((self.n_envs), dtype=np.int)
        self.d_neg_pos = np.ones((self.buffer_size, self.n_envs), dtype=np.bool_)

    def add_exp_data(self, exp_data):
        # save
        self.exp_data['obs'] = exp_data['state'].astype(np.float32)
        self.exp_data['next_obs'] = exp_data['next_state'].astype(np.float32)
        self.exp_data['action'] = exp_data['action'].astype(np.float32)
        self.exp_data['reward'] = exp_data['reward'].astype(np.float32)
        self.exp_data['cost'] = exp_data['cost'].astype(np.float32)
        self.exp_data['done'] = np.expand_dims(exp_data['done'].astype(np.float32), axis=1)

        # other init
        self.exp_len = exp_data['state'].shape[0] // self.n_envs
        self.pos = self.exp_len
        self.last_store_pos += self.pos

    def consider_exp_add(self, exp_buffer):
        self.exp_buffer = exp_buffer
        self.do_exp_add = True

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # if self.do_exp_add:
        #     add_ids = np.array([info['stage_success'] for info in infos])
        #     if np.sum(add_ids) > 0:
        #         self.exp_buffer.add(np.array(obs)[add_ids], np.array(action)[add_ids], np.array(done)[add_ids])
                # return

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.costs[self.pos] = np.array([info['cost'] for info in infos])
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = self.exp_len
        
        if self.do_exp_add:
            for env_id in range(len(infos)):
                # put into cache
                if infos[env_id]['state_store'] == 'keep':
                    continue
                # drop
                elif infos[env_id]['state_store'] == 'drop':
                    self.last_store_pos[env_id] = self.pos
                # store
                elif infos[env_id]['state_store'] == 'store':
                    # TODO: consider whether add whole trajectory or just success states
                    if self.last_store_pos[env_id] > self.pos:
                        # add_ids_after = np.where(self.rewards[self.last_store_pos[env_id]:, env_id] > 0)[0] + self.last_store_pos[env_id]
                        # add_ids_before = np.where(self.rewards[:self.pos, env_id] > 0)[0]
                        add_ids_after = np.arange(self.last_store_pos[env_id], self.rewards.shape[0])
                        add_ids_before = np.arange(0, self.pos)
                        add_ids = np.concatenate([add_ids_after, add_ids_before], axis=0)

                    else:
                        # add_ids = np.where(self.rewards[self.last_store_pos[env_id]: self.pos, env_id] > 0)[0]
                        add_ids = np.arange(self.last_store_pos[env_id], self.pos)

                    self.d_neg_pos[add_ids, env_id] = False
                    self.exp_buffer.add(self.observations[add_ids, env_id], self.actions[add_ids, env_id], self.dones[add_ids, env_id])
                    self.last_store_pos[env_id] = self.pos

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferLagSamples:
        # Sample from expert data is required
        exp_num = np.sum(batch_inds < self.exp_len)
        if exp_num > 0:
            # exp_ids = np.random.randint(0, high=self.exp_data['obs'].shape[0], size=exp_num)
            exp_ids = np.random.choice(np.arange(self.exp_data['obs'].shape[0]), size=exp_num, replace=False)
            exp_obs = self._normalize_obs(self.exp_data['obs'][exp_ids])
            exp_act = self.exp_data['action'][exp_ids]
            exp_next_obs = self._normalize_obs(self.exp_data['next_obs'][exp_ids])
            exp_dones = self.exp_data['done'][exp_ids]
            exp_reward = self._normalize_reward(self.exp_data['reward'][exp_ids].reshape(-1, 1), env)
            exp_cost = self._normalize_reward(self.exp_data['cost'][exp_ids].reshape(-1, 1), env)
            batch_inds = batch_inds[batch_inds >= self.exp_len]

        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = [
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self._normalize_reward(self.costs[batch_inds, env_indices].reshape(-1, 1), env)
        ]
        if exp_num > 0:
            data[0] = np.concatenate([exp_obs, data[0]], axis=0)
            data[1] = np.concatenate([exp_act, data[1]], axis=0)
            data[2] = np.concatenate([exp_next_obs, data[2]], axis=0)
            data[3] = np.concatenate([exp_dones, data[3]], axis=0)
            data[4] = np.concatenate([exp_reward, data[4]], axis=0)
            data[5] = np.concatenate([exp_cost, data[5]], axis=0)

        return ReplayBufferLagSamples(*tuple(map(self.to_torch, data)))

    def sample_neg(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # get valid_ids
        if self.full:
            valid_ids_0 = []
            valid_ids_1 = []
            for env_id in range(self.n_envs):
                if self.pos > self.last_store_pos[env_id]:
                    valid_ids_before = np.where(self.d_neg_pos[:self.last_store_pos[env_id], env_id])[0]
                    valid_ids_after = np.where(self.d_neg_pos[self.pos:, env_id])[0]
                    valid_ids_0.append(np.concatenate([valid_ids_before, valid_ids_after+self.pos]))
                    valid_ids_1.append(np.zeros_like(valid_ids_0[-1]) + env_id)
                else:
                    valid_ids = np.where(self.d_neg_pos[self.pos: self.last_store_pos[env_id], env_id])[0]
                    valid_ids_0.append(valid_ids[0] + self.pos)
                    valid_ids_1.append(np.zeros_like(valid_ids_0[-1]) + env_id)
            valid_ids = (np.concatenate(valid_ids_0), np.concatenate(valid_ids_1))
        else:
            valid_ids_0 = []
            valid_ids_1 = []
            for env_id in range(self.n_envs):
                valid_ids_0.append(np.where(self.d_neg_pos[:self.last_store_pos[env_id], env_id])[0])
                valid_ids_1.append(np.zeros_like(valid_ids_0[-1]) + env_id)

            valid_ids = (np.concatenate(valid_ids_0), np.concatenate(valid_ids_1))

        # pick
        pick_idx = np.random.choice(np.arange(len(valid_ids[0])), batch_size, replace=False)
        env_pick_valid_ids = (valid_ids[0][pick_idx],
                              valid_ids[1][pick_idx])

        # get data
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(env_pick_valid_ids[0]+1, env_pick_valid_ids[1])], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[env_pick_valid_ids], env)
        
        data = [self._normalize_obs(self.observations[env_pick_valid_ids], env),
                self.actions[env_pick_valid_ids],
                next_obs,
                # Only use dones that are not due to timeouts
                # deactivated by default (timeouts is initialized as an array of False)
                (self.dones[env_pick_valid_ids] * (1 - self.timeouts[env_pick_valid_ids])).reshape(-1, 1),
                self._normalize_reward(self.rewards[env_pick_valid_ids].reshape(-1, 1), env),
                self._normalize_reward(self.costs[env_pick_valid_ids].reshape(-1, 1), env)
        ]

        return ReplayBufferLagSamples(*tuple(map(self.to_torch, data)))

class ReplayBufferSimpleBag(ReplayBuffer):
    costs: np.ndarray

    def __init__(self,
                buffer_size: int,
                observation_space: spaces.Space,
                action_space: spaces.Space,
                device: Union[th.device, str] = "auto",
                n_envs: int = 1,
                optimize_memory_usage: bool = False,
                handle_timeout_termination: bool = True,
                policy_num: int = 1,
                replay_buffer_dict=None):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)
        self.replay_buffer_dict = replay_buffer_dict
        self.policy_num = policy_num

    def check_available(self, policy_id, batch_size=0):
        return self.replay_buffer_dict[str(policy_id)].size() > batch_size


    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        assert self.n_envs == 1
        policy_id = infos[0]['policy_id']
        done = np.logical_or(done, np.array([info['stage_done'] for info in infos]))
        self.replay_buffer_dict[str(policy_id)].add(obs, next_obs, action, reward, done, infos)


    def sample(self, batch_size: int, policy_id: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        return self.replay_buffer_dict[str(policy_id)].sample(batch_size, env)


class ReplayBufferLagBag(ReplayBuffer):
    costs: np.ndarray

    def __init__(self,
                buffer_size: int,
                observation_space: spaces.Space,
                action_space: spaces.Space,
                device: Union[th.device, str] = "auto",
                n_envs: int = 1,
                optimize_memory_usage: bool = False,
                handle_timeout_termination: bool = True,
                policy_num: int = 1):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.last_store_pos = np.zeros((self.n_envs), dtype=np.int)
        self.d_neg_pos = np.ones((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.policy_ids = np.zeros((self.buffer_size, self.n_envs, policy_num), dtype=np.bool_)
        self.policy_num = policy_num
        self.reward_record = {policy_id: [[0, 0] for _ in range(self.n_envs)] for policy_id in range(policy_num)}
        # self.exp_data_dict = {'obs': None, 'next_obs': None, 'action': None, 'reward': None, 'cost': None, 'done': None, 'policy_ids': None}
        self.exp_data_dict = {}
        self.exp_len_dict = {}
        # self.exp_len_dict = {policy_id:0 for policy_id in range(policy_num)}
        self.do_exp_add = False
        self.start_pos = 0

    def check_available(self, policy_id, batch_size=0):
        exp_len = self.exp_len_dict[policy_id] if policy_id in self.exp_len_dict else 0
        if self.full:
            return np.sum(self.policy_ids[:, :, policy_id]) + exp_len > batch_size
        else:
            return np.sum(self.policy_ids[:self.pos, :, policy_id]) + exp_len > batch_size

    def get_stage_reward(self):
        ave_reward = {policy_id:0 for policy_id in range(self.policy_num)}
        for policy_id in range(self.policy_num):
            for env_id in range(self.n_envs):
                if len(self.reward_record[policy_id][env_id]) > 1:
                    ave_reward[policy_id] += self.reward_record[policy_id][env_id][-2]
                else:
                    ave_reward[policy_id] += self.reward_record[policy_id][env_id][-1]
            ave_reward[policy_id] = ave_reward[policy_id] / self.n_envs
        
        return ave_reward

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.costs[self.pos] = np.array([info['cost'] for info in infos])

        self.dones[self.pos] = np.logical_or(np.array(done), np.array([info['stage_done'] for info in infos]))

        # record reward for stage done (although not recommended do it here)
        for env_id in range(self.n_envs):
            self.reward_record[infos[env_id]['policy_id']][env_id][-1] += reward[env_id]
            if done[env_id] or infos[env_id]['stage_done']:
                for policy_id in range(self.policy_num):
                    if self.reward_record[policy_id][env_id][-1] != 0:
                        self.reward_record[policy_id][env_id][-2] = self.reward_record[policy_id][env_id][-1]
                        self.reward_record[policy_id][env_id][-1] = 0

        # add policy number
        cur_policy_ids = np.expand_dims(np.array([info['policy_id'] for info in infos]), axis=1)
        idxs = np.expand_dims(np.arange(len(infos)), axis=1)
        self.policy_ids[self.pos][idxs, cur_policy_ids] = True

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            # self.pos = 0
            self.pos = self.start_pos
        
        if self.do_exp_add:
            for env_id in range(len(infos)):
                # put into cache
                if infos[env_id]['state_store'] == 'keep':
                    continue
                # drop
                elif infos[env_id]['state_store'] == 'drop':
                    self.last_store_pos[env_id] = self.pos
                # store
                elif infos[env_id]['state_store'] == 'store':
                    # TODO: consider whether add whole trajectory or just success states
                    if self.last_store_pos[env_id] > self.pos:
                        # add_ids_after = np.where(self.rewards[self.last_store_pos[env_id]:, env_id] > 0)[0] + self.last_store_pos[env_id]
                        # add_ids_before = np.where(self.rewards[:self.pos, env_id] > 0)[0]
                        if self.pos < self.last_store_pos[env_id]:
                            add_ids_after = np.arange(self.last_store_pos[env_id], self.rewards.shape[0])
                            add_ids_before = np.arange(0, self.pos)
                            add_ids = np.concatenate([add_ids_after, add_ids_before], axis=0)
                        else:
                            add_ids = np.arange(self.last_store_pos[env_id], self.pos)

                    else:
                        # add_ids = np.where(self.rewards[self.last_store_pos[env_id]: self.pos, env_id] > 0)[0]
                        add_ids = np.arange(self.last_store_pos[env_id], self.pos)

                    self.d_neg_pos[add_ids, env_id] = False
                    for cur_policy_id in range(self.policy_num):
                        cur_ids = np.where(self.policy_ids[add_ids, env_id, cur_policy_id])[0]
                        if len(cur_ids) > 0:
                            self.exp_buffer_dict[cur_policy_id].add(self.observations[add_ids[cur_ids], env_id], self.actions[add_ids[cur_ids], env_id], self.dones[add_ids[cur_ids], env_id])
                            self.last_store_pos[env_id] = self.pos

                            # only for debug
                            # self.rewards[add_ids[cur_ids], env_id] += 1

    def add_exp_data(self, exp_data_dict):
        # init
        self.exp_data_dict = {}
        self.exp_len_dict = {}
        max_exp_len = 0
        for data_id in exp_data_dict:
            # save
            self.exp_data_dict[data_id] = {}
            self.exp_data_dict[data_id]['obs'] = exp_data_dict[data_id]['state'].astype(np.float32)
            self.exp_data_dict[data_id]['next_obs'] = exp_data_dict[data_id]['next_state'].astype(np.float32)
            self.exp_data_dict[data_id]['action'] = exp_data_dict[data_id]['action'].astype(np.float32)
            self.exp_data_dict[data_id]['reward'] = exp_data_dict[data_id]['reward'].astype(np.float32)
            self.exp_data_dict[data_id]['cost'] = exp_data_dict[data_id]['cost'].astype(np.float32)
            self.exp_data_dict[data_id]['done'] = np.expand_dims(exp_data_dict[data_id]['done'].astype(np.float32), axis=1)

            # other init
            self.exp_len_dict[data_id] = exp_data_dict[data_id]['state'].shape[0]
            if self.exp_len_dict[data_id] > max_exp_len:
                max_exp_len = self.exp_len_dict[data_id]

        self.pos = max_exp_len
        # self.start_pos = max_exp_len
        self.start_pos = 0
        self.last_store_pos += self.pos

    def consider_exp_add(self, exp_buffer_dict):
        self.exp_buffer_dict = exp_buffer_dict
        self.do_exp_add = True

    def sample(self, batch_size: int, policy_id: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # get valid_ids
        if self.full:
            valid_ids = np.where(self.policy_ids[:, :, policy_id])
        else:
            valid_ids = np.where(self.policy_ids[:self.pos, :, policy_id])

        # only for debug
        assert np.sum(self.policy_ids[:self.start_pos, :, policy_id]) == 0

        valid_num = valid_ids[0].shape[0]
        exp_len = self.exp_len_dict[policy_id] if policy_id in self.exp_len_dict else 0
        valid_num += exp_len

        # pick id (must have enough example)
        assert valid_num >= batch_size
        pick_idx = np.random.choice(np.arange(valid_num), batch_size, replace=False)

        # Sample from expert data if required
        if policy_id in self.exp_len_dict:
            exp_data = self.exp_data_dict[policy_id]
            exp_num = np.sum(pick_idx < exp_len)
            if exp_num > 0:
                exp_pick_valid_ids = np.random.choice(np.arange(exp_data['obs'].shape[0]), size=exp_num, replace=False)
                exp_obs = self._normalize_obs(exp_data['obs'][exp_pick_valid_ids])
                exp_act = exp_data['action'][exp_pick_valid_ids]
                exp_next_obs = self._normalize_obs(exp_data['next_obs'][exp_pick_valid_ids])
                exp_dones = exp_data['done'][exp_pick_valid_ids]
                exp_reward = self._normalize_reward(exp_data['reward'][exp_pick_valid_ids].reshape(-1, 1), env)
                exp_cost = self._normalize_reward(exp_data['cost'][exp_pick_valid_ids].reshape(-1, 1), env)

        # Sample randomly from the env idx
        env_pick_valid_ids = (valid_ids[0][pick_idx[pick_idx >= exp_len] - exp_len],
                              valid_ids[1][pick_idx[pick_idx >= exp_len] - exp_len])

        # get data
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(env_pick_valid_ids[0]+1, env_pick_valid_ids[1])], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[env_pick_valid_ids], env)
        
        data = [self._normalize_obs(self.observations[env_pick_valid_ids], env),
                self.actions[env_pick_valid_ids],
                next_obs,
                # Only use dones that are not due to timeouts
                # deactivated by default (timeouts is initialized as an array of False)
                (self.dones[env_pick_valid_ids] * (1 - self.timeouts[env_pick_valid_ids])).reshape(-1, 1),
                self._normalize_reward(self.rewards[env_pick_valid_ids].reshape(-1, 1), env),
                self._normalize_reward(self.costs[env_pick_valid_ids].reshape(-1, 1), env)
        ]
        if policy_id in self.exp_len_dict and exp_num > 0:
            data[0] = np.concatenate([exp_obs, data[0]], axis=0)
            data[1] = np.concatenate([exp_act, data[1]], axis=0)
            data[2] = np.concatenate([exp_next_obs, data[2]], axis=0)
            data[3] = np.concatenate([exp_dones, data[3]], axis=0)
            data[4] = np.concatenate([exp_reward, data[4]], axis=0)
            data[5] = np.concatenate([exp_cost, data[5]], axis=0)

        return ReplayBufferLagSamples(*tuple(map(self.to_torch, data)))
    
    def _sample_neg_get_valid(self, policy_ids, end_ids, cur_policy_id):
        valid_ids_0 = []
        valid_ids_1 = []
        for env_id in range(self.n_envs):
            valid_ids_0.append(np.where(np.logical_and(policy_ids[:end_ids[env_id], env_id, cur_policy_id], \
                                                       self.d_neg_pos[:end_ids[env_id], env_id]))[0])
            valid_ids_1.append(np.zeros_like(valid_ids_0[-1]) + env_id)

        return (np.concatenate(valid_ids_0), np.concatenate(valid_ids_1))


    def sample_neg(self, batch_size: int, policy_id: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # get valid_ids
        if self.full:
            valid_ids_0 = []
            valid_ids_1 = []
            for env_id in range(self.n_envs):
                if self.pos > self.last_store_pos[env_id]:
                    valid_ids_before = np.where(np.logical_and(self.policy_ids[:self.last_store_pos[env_id], env_id, policy_id],\
                                                               self.d_neg_pos[:self.last_store_pos[env_id], env_id]))[0]
                    valid_ids_after = np.where(np.logical_and(self.policy_ids[self.pos:, env_id, policy_id],\
                                                              self.d_neg_pos[self.pos:, env_id]))[0]
                    valid_ids_0.append(np.concatenate([valid_ids_before, valid_ids_after+self.pos]))
                    valid_ids_1.append(np.zeros_like(valid_ids_0[-1]) + env_id)
                else:
                    valid_ids = np.where(np.logical_and(self.policy_ids[self.pos: self.last_store_pos[env_id], env_id, policy_id],\
                                                        self.d_neg_pos[self.pos: self.last_store_pos[env_id], env_id]))[0]
                    valid_ids_0.append(valid_ids[0] + self.pos)
                    valid_ids_1.append(np.zeros_like(valid_ids_0[-1]) + env_id)
            valid_ids = (np.concatenate(valid_ids_0), np.concatenate(valid_ids_1))
        else:
            # valid_ids = np.where(self.policy_ids[:self.last_store_pos[policy_id], :, policy_id])
            valid_ids = self._sample_neg_get_valid(self.policy_ids, self.last_store_pos, policy_id)

        # only for debug
        assert np.sum(self.policy_ids[:self.start_pos, :, policy_id]) == 0

        # pick
        pick_idx = np.random.choice(np.arange(len(valid_ids[0])), batch_size, replace=False)
        env_pick_valid_ids = (valid_ids[0][pick_idx],
                              valid_ids[1][pick_idx])

        # get data
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(env_pick_valid_ids[0]+1, env_pick_valid_ids[1])], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[env_pick_valid_ids], env)
        
        data = [self._normalize_obs(self.observations[env_pick_valid_ids], env),
                self.actions[env_pick_valid_ids],
                next_obs,
                # Only use dones that are not due to timeouts
                # deactivated by default (timeouts is initialized as an array of False)
                (self.dones[env_pick_valid_ids] * (1 - self.timeouts[env_pick_valid_ids])).reshape(-1, 1),
                self._normalize_reward(self.rewards[env_pick_valid_ids].reshape(-1, 1), env),
                self._normalize_reward(self.costs[env_pick_valid_ids].reshape(-1, 1), env)
        ]

        return ReplayBufferLagSamples(*tuple(map(self.to_torch, data)))

    def get_spec_num(self):
        policy_num_dict = {}
        for policy_id in range(self.policy_num):
            policy_num_dict[policy_id] = np.sum(self.policy_ids[:self.pos, :, policy_id])

        return policy_num_dict
    

class ReplayBufferLagBagDebug(ReplayBuffer):
    costs: np.ndarray

    def __init__(self,
                buffer_size: int,
                observation_space: spaces.Space,
                action_space: spaces.Space,
                device: Union[th.device, str] = "auto",
                n_envs: int = 1,
                optimize_memory_usage: bool = False,
                handle_timeout_termination: bool = True,
                policy_num: int = 1):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.last_store_pos = np.zeros((self.n_envs), dtype=np.int)
        self.d_neg_pos = np.ones((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.policy_ids = np.zeros((self.buffer_size, self.n_envs, policy_num), dtype=np.bool_)
        self.policy_num = policy_num
        self.reward_record = {policy_id: [[0, 0] for _ in range(self.n_envs)] for policy_id in range(policy_num)}
        self.exp_data_dict = {}
        self.exp_len_dict = {}
        self.do_exp_add = False
        self.start_pos = 0

    def check_available(self, policy_id, batch_size=0):
        if self.full:
            return np.sum(self.policy_ids[:, :, policy_id]) + self.exp_len_dict[policy_id] > batch_size
        else:
            return np.sum(self.policy_ids[:self.pos, :, policy_id]) + self.exp_len_dict[policy_id] > batch_size

    def get_stage_reward(self):
        ave_reward = {policy_id:0 for policy_id in range(self.policy_num)}
        for policy_id in range(self.policy_num):
            for env_id in range(self.n_envs):
                if len(self.reward_record[policy_id][env_id]) > 1:
                    ave_reward[policy_id] += self.reward_record[policy_id][env_id][-2]
                else:
                    ave_reward[policy_id] += self.reward_record[policy_id][env_id][-1]
            ave_reward[policy_id] = ave_reward[policy_id] / self.n_envs
        
        return ave_reward

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.costs[self.pos] = np.array([info['cost'] for info in infos])

        self.dones[self.pos] = np.logical_or(np.array(done), np.array([info['stage_done'] for info in infos]))

        # record reward for stage done (although not recommended do it here)
        for env_id in range(self.n_envs):
            self.reward_record[infos[env_id]['policy_id']][env_id][-1] += reward[env_id]
            if done[env_id] or infos[env_id]['stage_done']:
                for policy_id in range(self.policy_num):
                    if self.reward_record[policy_id][env_id][-1] != 0:
                        self.reward_record[policy_id][env_id][-2] = self.reward_record[policy_id][env_id][-1]
                        self.reward_record[policy_id][env_id][-1] = 0

        # add policy number
        cur_policy_ids = np.expand_dims(np.array([info['policy_id'] for info in infos]), axis=1)
        idxs = np.expand_dims(np.arange(len(infos)), axis=1)
        self.policy_ids[self.pos][idxs, cur_policy_ids] = True

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            # self.pos = 0
            self.pos = self.start_pos
        
        if self.do_exp_add:
            for env_id in range(len(infos)):
                # put into cache
                if infos[env_id]['state_store'] == 'keep':
                    continue
                # drop
                elif infos[env_id]['state_store'] == 'drop':
                    self.last_store_pos[env_id] = self.pos
                # store
                elif infos[env_id]['state_store'] == 'store':
                    # TODO: consider whether add whole trajectory or just success states
                    if self.last_store_pos[env_id] > self.pos:
                        # add_ids_after = np.where(self.rewards[self.last_store_pos[env_id]:, env_id] > 0)[0] + self.last_store_pos[env_id]
                        # add_ids_before = np.where(self.rewards[:self.pos, env_id] > 0)[0]
                        if self.pos < self.last_store_pos[env_id]:
                            add_ids_after = np.arange(self.last_store_pos[env_id], self.rewards.shape[0])
                            add_ids_before = np.arange(0, self.pos)
                            add_ids = np.concatenate([add_ids_after, add_ids_before], axis=0)
                        else:
                            add_ids = np.arange(self.last_store_pos[env_id], self.pos)

                    else:
                        # add_ids = np.where(self.rewards[self.last_store_pos[env_id]: self.pos, env_id] > 0)[0]
                        add_ids = np.arange(self.last_store_pos[env_id], self.pos)

                    self.d_neg_pos[add_ids, env_id] = False
                    for cur_policy_id in range(self.policy_num):
                        cur_ids = np.where(self.policy_ids[add_ids, env_id, cur_policy_id])[0]
                        if len(cur_ids) > 0:
                            self.exp_buffer_dict[cur_policy_id].add(self.observations[add_ids[cur_ids], env_id], self.actions[add_ids[cur_ids], env_id], self.dones[add_ids[cur_ids], env_id])
                            self.last_store_pos[env_id] = self.pos

    def add_exp_data(self, exp_data_dict):
        # init
        self.exp_data_dict = {}
        self.exp_len_dict = {}
        max_exp_len = 0
        for data_id in exp_data_dict:
            # save
            self.exp_data_dict[data_id] = {}
            self.exp_data_dict[data_id]['obs'] = exp_data_dict[data_id]['state'].astype(np.float32)
            self.exp_data_dict[data_id]['next_obs'] = exp_data_dict[data_id]['next_state'].astype(np.float32)
            self.exp_data_dict[data_id]['action'] = exp_data_dict[data_id]['action'].astype(np.float32)
            self.exp_data_dict[data_id]['reward'] = exp_data_dict[data_id]['reward'].astype(np.float32)
            self.exp_data_dict[data_id]['cost'] = exp_data_dict[data_id]['cost'].astype(np.float32)
            self.exp_data_dict[data_id]['done'] = np.expand_dims(exp_data_dict[data_id]['done'].astype(np.float32), axis=1)

            # other init
            self.exp_len_dict[data_id] = exp_data_dict[data_id]['state'].shape[0]
            if self.exp_len_dict[data_id] > max_exp_len:
                max_exp_len = self.exp_len_dict[data_id]

        self.pos = max_exp_len
        # self.start_pos = max_exp_len
        self.start_pos = 0
        self.last_store_pos += self.pos

    def consider_exp_add(self, exp_buffer_dict):
        self.exp_buffer_dict = exp_buffer_dict
        self.do_exp_add = True

    def sample(self, batch_size: int, policy_id: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # get valid_ids
        if self.full:
            valid_ids = np.where(self.policy_ids[:, :, policy_id])
        else:
            valid_ids = np.where(self.policy_ids[:self.pos, :, policy_id])

        # only for debug
        assert np.sum(self.policy_ids[:self.start_pos, :, policy_id]) == 0

        valid_num = valid_ids[0].shape[0]
        exp_len = self.exp_len_dict[policy_id] if policy_id in self.exp_len_dict else 0
        valid_num += exp_len

        # pick id (must have enough example)
        assert valid_num >= batch_size
        pick_idx = np.random.choice(np.arange(valid_num), batch_size, replace=False)

        # Sample from expert data if required
        if policy_id in self.exp_len_dict:
            exp_data = self.exp_data_dict[policy_id]
            exp_num = np.sum(pick_idx < exp_len)
            if exp_num > 0:
                exp_pick_valid_ids = np.random.choice(np.arange(exp_data['obs'].shape[0]), size=exp_num, replace=False)
                exp_obs = self._normalize_obs(exp_data['obs'][exp_pick_valid_ids])
                exp_act = exp_data['action'][exp_pick_valid_ids]
                exp_next_obs = self._normalize_obs(exp_data['next_obs'][exp_pick_valid_ids])
                exp_dones = exp_data['done'][exp_pick_valid_ids]
                exp_reward = self._normalize_reward(exp_data['reward'][exp_pick_valid_ids].reshape(-1, 1), env)
                exp_cost = self._normalize_reward(exp_data['cost'][exp_pick_valid_ids].reshape(-1, 1), env)

        # Sample randomly from the env idx
        env_pick_valid_ids = (valid_ids[0][pick_idx[pick_idx >= exp_len] - exp_len],
                              valid_ids[1][pick_idx[pick_idx >= exp_len] - exp_len])

        # get data
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(env_pick_valid_ids[0]+1, env_pick_valid_ids[1])], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[env_pick_valid_ids], env)
        
        data = [self._normalize_obs(self.observations[env_pick_valid_ids], env),
                self.actions[env_pick_valid_ids],
                next_obs,
                # Only use dones that are not due to timeouts
                # deactivated by default (timeouts is initialized as an array of False)
                (self.dones[env_pick_valid_ids] * (1 - self.timeouts[env_pick_valid_ids])).reshape(-1, 1),
                self._normalize_reward(self.rewards[env_pick_valid_ids].reshape(-1, 1), env),
                self._normalize_reward(self.costs[env_pick_valid_ids].reshape(-1, 1), env)
        ]
        if exp_num > 0:
            data[0] = np.concatenate([exp_obs, data[0]], axis=0)
            data[1] = np.concatenate([exp_act, data[1]], axis=0)
            data[2] = np.concatenate([exp_next_obs, data[2]], axis=0)
            data[3] = np.concatenate([exp_dones, data[3]], axis=0)
            data[4] = np.concatenate([exp_reward, data[4]], axis=0)
            data[5] = np.concatenate([exp_cost, data[5]], axis=0)

        return ReplayBufferLagSamples(*tuple(map(self.to_torch, data)))

    def _sample_neg_get_valid(self, policy_ids, end_ids, cur_policy_id):
        valid_ids_0 = []
        valid_ids_1 = []
        for env_id in range(self.n_envs):
            valid_ids_0.append(np.where(np.logical_and(policy_ids[:end_ids[env_id], env_id, cur_policy_id], \
                                                       self.d_neg_pos[:end_ids[env_id], env_id]))[0])
            valid_ids_1.append(np.zeros_like(valid_ids_0[-1]) + env_id)

        return (np.concatenate(valid_ids_0), np.concatenate(valid_ids_1))


    def sample_neg(self, batch_size: int, policy_id: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # get valid_ids
        if self.full:
            valid_ids_0 = []
            valid_ids_1 = []
            for env_id in range(self.n_envs):
                if self.pos > self.last_store_pos[env_id]:
                    valid_ids_before = np.where(np.logical_and(self.policy_ids[:self.last_store_pos[env_id], env_id, policy_id],\
                                                               self.d_neg_pos[:self.last_store_pos[env_id], env_id]))[0]
                    valid_ids_after = np.where(np.logical_and(self.policy_ids[self.pos:, env_id, policy_id],\
                                                              self.d_neg_pos[self.pos:, env_id]))[0]
                    valid_ids_0.append(np.concatenate([valid_ids_before, valid_ids_after+self.pos]))
                    valid_ids_1.append(np.zeros_like(valid_ids_0[-1]) + env_id)
                else:
                    valid_ids = np.where(np.logical_and(self.policy_ids[self.pos: self.last_store_pos[env_id], env_id, policy_id],\
                                                        self.d_neg_pos[self.pos: self.last_store_pos[env_id], env_id]))[0]
                    valid_ids_0.append(valid_ids[0] + self.pos)
                    valid_ids_1.append(np.zeros_like(valid_ids_0[-1]) + env_id)
            valid_ids = (np.concatenate(valid_ids_0), np.concatenate(valid_ids_1))
        else:
            # valid_ids = np.where(self.policy_ids[:self.last_store_pos[policy_id], :, policy_id])
            valid_ids = self._sample_neg_get_valid(self.policy_ids, self.last_store_pos, policy_id)

        # pdb.set_trace()

        # only for debug
        assert np.sum(self.policy_ids[:self.start_pos, :, policy_id]) == 0

        # pick
        pick_idx = np.random.choice(np.arange(len(valid_ids[0])), batch_size, replace=False)
        env_pick_valid_ids = (valid_ids[0][pick_idx],
                              valid_ids[1][pick_idx])

        # get data
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(env_pick_valid_ids[0]+1, env_pick_valid_ids[1])], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[env_pick_valid_ids], env)
        
        data = [self._normalize_obs(self.observations[env_pick_valid_ids], env),
                self.actions[env_pick_valid_ids],
                next_obs,
                # Only use dones that are not due to timeouts
                # deactivated by default (timeouts is initialized as an array of False)
                (self.dones[env_pick_valid_ids] * (1 - self.timeouts[env_pick_valid_ids])).reshape(-1, 1),
                self._normalize_reward(self.rewards[env_pick_valid_ids].reshape(-1, 1), env),
                self._normalize_reward(self.costs[env_pick_valid_ids].reshape(-1, 1), env)
        ]

        return ReplayBufferLagSamples(*tuple(map(self.to_torch, data)))


    def get_spec_num(self):
        policy_num_dict = {}
        for policy_id in range(self.policy_num):
            policy_num_dict[policy_id] = np.sum(self.policy_ids[:self.pos, :, policy_id])

        return policy_num_dict