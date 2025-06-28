import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import os
import math

import matplotlib.pyplot as plt

from modules.reward import RewIdentity, RewDebug, RewCritic
from utils.logging import init_logging, log_and_print

import pdb

def common_skill_reward(reward, cost, rew_fun, lagrangian):
    if not isinstance(rew_fun, RewIdentity):
        reward -= 0
        if lagrangian:
            if reward >= 0 and cost <= 0:
                reward += 1
        elif reward >= 0:
            reward += 1
    else:
        if reward == -1:
            reward = -1
        elif reward == 0:
            reward = 1
        else:
            print(reward)
            pdb.set_trace()
            raise NotImplementedError
        
    return reward

def block_skill_reward(reward, cost, rew_fun, lagrangian):
    if not isinstance(rew_fun, RewIdentity):
        # only for debug
        reward -= 0
        if lagrangian:
            if reward >= 0 and cost <= 0:
                reward += 1
        elif reward >= 0:
            reward += 1
    else:
        if reward == -1:
            reward = -1
        elif reward == 0:
            reward = 0
        elif reward == 1:
            reward = 1
        else:
            print(reward)
            pdb.set_trace()
            raise NotImplementedError
        
    return reward

class SkillEnv(gym.Wrapper):
    def __init__(self, env, skill_graph, search_node, rew_fun, threshold=1.0, traj_len=100, hold_len=1, train=False, lagrangian=False, \
                 fail_search=False, additional_reward_fun=common_skill_reward, task_limit=None):
        super().__init__(env)
        self.attemp_num = 1000
        self.skill_graph = skill_graph
        self.search_node = search_node
        self.rew_fun = rew_fun
        self.env_success_rew = env.env_success_rew
        self.additional_reward_fun = additional_reward_fun
        self.done_thre = threshold
        self.traj_len = traj_len
        self.traj_id = 0
        self.train = train
        self.task_limit_mode = task_limit is not None
        self.task_limit = np.inf if task_limit is None else task_limit
        self.task_step = 0
        self.tanh_fun = nn.Tanh()

        # should be no less than 1
        self.hold_len = hold_len
        self.hold_step = 0
        assert self.hold_len >= 1

        self.debug_id = 0
        self.reset_time = 0
        self.lagrangian = lagrangian
        self.fail_search = fail_search

        # get all reward functions
        self.stage_rewards = {}
        cur_node = self.skill_graph.start_node.s_node
        cur_state_rew = 1
        while cur_node is not None:
            self.stage_rewards[cur_state_rew] = cur_node.reward_fun[-1]
            cur_node = cur_node.s_node
            cur_state_rew += 1
        self.max_state_rew = cur_state_rew

    def set_train(self):
        self.rew_fun.set_train()
        self.train = True

    def set_eval(self):
        self.rew_fun.set_eval()
        self.train = False

    def reset(self, **kwargs):
        # reset
        obs, info = self.env.reset(**kwargs)
        self.traj_id = 0
        self.task_step = 0
        self.hold_step = 0

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['cost'] = 0
        self.traj_id += 1
        self.task_step += 1

        # environment success
        if reward >= 0:
            truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
            if self.train:
                terminated = False
            else:
                terminated = True
            new_reward = self.max_state_rew
            return obs, new_reward, terminated, truncated, info
        else:
            # check which one satisfied
            state_reward = 0
            for new_state_reward in self.stage_rewards:
                cur_rew_fun = self.stage_rewards[new_state_reward]
                cur_rew_fun.set_eval()
                if cur_rew_fun.get_reward(obs, reward):
                    state_reward = new_state_reward
            if state_reward == self.max_state_rew-1:
                truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
                if self.train:
                    terminated = False
                else:
                    terminated = True
                new_reward = self.max_state_rew
                return obs, new_reward, terminated, truncated, info
            # assert state_reward != self.max_state_rew-1
            applied_rew_fun = self.stage_rewards[state_reward+1]

        # reward in train mode
        applied_rew_fun.set_train()
        if self.lagrangian and not isinstance(applied_rew_fun, RewIdentity) and self.train:
            train_reward, train_cost = applied_rew_fun.get_reward(obs, reward)
            train_reward = 0.5 * self.tanh_fun(torch.tensor(10 * train_reward)).item() - \
                           0.5 * self.tanh_fun(torch.tensor(10 * train_cost)).item() + state_reward
        else:
            train_reward = applied_rew_fun.get_reward(obs, reward)
            train_reward = 0.5 * self.tanh_fun(torch.tensor(10 * train_reward)) + state_reward

        truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
        return obs, train_reward, False, truncated, info