import gymnasium as gym
import numpy as np
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
                 fail_search=False, additional_reward_fun=common_skill_reward, task_limit=None, env_debug=False):
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

        # should be no less than 1
        self.hold_len = hold_len
        self.hold_step = 0
        assert self.hold_len >= 1

        self.debug_id = 0
        self.reset_time = 0
        self.lagrangian = lagrangian
        self.fail_search = fail_search

        self.env_debug = env_debug
        if self.env_debug:
            self.observation_space = gym.spaces.Box(-math.inf, math.inf, shape=[28])

    def set_train(self):
        self.rew_fun.set_train()
        self.train = True

    def set_eval(self):
        self.rew_fun.set_eval()
        self.train = False

    def reset(self, **kwargs):
        # loop
        cur_attempt = 0
        self.hold_step = 0

        # only for debug
        drop_num = 0

        while True:
            self.traj_id = 0

            # obs, info, success, traj_len = self.skill_graph.rollout(self.env, self.traj_len, self.search_node.node_id, drop_success=False, fail_search=self.fail_search, **kwargs)
            obs, info, success, traj_len = self.skill_graph.rollout(self.env, self.traj_len, self.search_node.node_id, \
                                                                    drop_success=True, fail_search=self.fail_search, traj_limit_mode=self.task_limit_mode, **kwargs)
            self.traj_id += traj_len
            if info['drop']:
                drop_num += 1

            # success
            if success:
                break
            else:
                cur_attempt += 1
                # print('{}/{}'.format(drop_num, cur_attempt))
                if cur_attempt > self.attemp_num:
                    # print(drop_num)
                    raise Exception('reset fail on skill environment')

        self.reset_time += 1

        if self.env_debug:
            obs = self.env.env.get_custom_obs(0)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.lagrangian and not isinstance(self.rew_fun, RewIdentity) and self.train:
            new_reward, new_cost = self.rew_fun.get_reward(obs, reward)
            info['cost'] = new_cost
        else:
            new_reward = self.rew_fun.get_reward(obs, reward)
            new_cost = None
            info['cost'] = 0
        self.traj_id += 1
        self.task_step += 1

        # train mode
        if self.train:
            truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
            # penalty
            new_reward = self.additional_reward_fun(new_reward, new_cost, self.rew_fun, self.lagrangian)

            if self.env_debug:
                obs = self.env.env.get_custom_obs(0)

            if truncated:
                return obs, new_reward, False, truncated, info

            # if success, rollout to get next
            if new_reward > 0 and new_cost >= 0:
                self.hold_step += 1
            else:
                self.hold_step = 0
            if self.hold_step >= self.hold_len:
                self.hold_step = 0
                obs, _, success, _ = self.skill_graph.rollout(self.env, self.traj_len-self.traj_id, self.search_node.node_id, \
                                                                        drop_success=True, fail_search=self.fail_search, traj_limit_mode=self.task_limit_mode)
                if not success:
                    truncated = True

            return obs, new_reward, False, truncated, info

        # eval mode
        else:
            # environment success
            if reward >= 0:
                truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
                terminated = True
                new_reward = 1
            else:
                if new_reward >= self.done_thre:
                    self.hold_step += 1
                else:
                    self.hold_step = 0
                truncated = self.traj_id >= self.traj_len or self.task_step >= self.task_limit
                terminated = self.hold_step >= self.hold_len

            if self.env_debug:
                obs = self.env.env.get_custom_obs(0)

            return obs, new_reward, terminated, truncated, info

class CustomMonitor(gym.Wrapper):
    def __init__(self, env, log_freq, rew_fun):
        super().__init__(env)
        self.log_freq = log_freq
        self.reset_time = 0
        self.rew_fun = rew_fun
        self.rew_store = []
        self.act_store = []
        self.img_store = []

    def reset(self, **kwargs):
        self.reset_time += 1
        if self.reset_time % self.log_freq == 0:
            print('do store')
            self.rew_store = []
            self.act_store = []
            self.img_store = []
            # self.rew_store.append([])
            # self.act_store.append([])

        obs, info = self.env.reset(**kwargs)
        if self.reset_time % self.log_freq == 0:
            # self.rew_store.append(self.env.get_reward())
            self.img_store.append(self.env.render())

        if self.reset_time % self.log_freq == 1 and self.reset_time!=1:
            print('print')
            # plt.figure()
            # rew_data = self.rew_store
            # plt.plot(np.arange(len(rew_data)), rew_data, 'k-')
            # plt.savefig('store/tree_push_debug_4/debug_fig/rew_{}.png'.format(self.reset_time*100))
            # plt.close()

            plt.figure()
            action_data = np.array(self.act_store)
            plt.plot(np.arange(len(action_data)), action_data[:, -1], 'k-')
            plt.savefig('store/tree_push_debug_3/debug_fig/act_{}.png'.format(self.reset_time*100))
            plt.close()

            os.makedirs('store/tree_push_debug_3/debug_fig/{}'.format(self.reset_time*100))
            plt.figure()
            for img_id, img in enumerate(self.img_store):
                plt.imshow(img)
                plt.savefig('store/tree_push_debug_3/debug_fig/{}/{}.png'.format(self.reset_time*100, img_id))
                plt.cla()
            plt.close()

        return obs, info
        
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.reset_time % self.log_freq == 0:
            # dense_result = self.rew_fun.equ.execute_dense_details(np.expand_dims(obs, axis=0))
            dense_result = reward
            self.rew_store.append(dense_result)
            self.act_store.append(action)
            self.img_store.append(self.env.render())

        return obs, reward, terminated, truncated, info

class AbsTransit_Pick:
    def __init__(self):
        self.add_dim = 3

    def get_abs_obs(self, obs):
        # distance between objects
        gripper_block_dist, goal_block_dist, gripper_goal_dist, gripper_dist = self.get_distance(obs)
        gripper_block_z_dist, goal_block_z_dist, goal_gripper_z_dist = self.get_z_distance(obs)

        return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
                               gripper_dist], \
                            axis=1)
        # return np.concatenate([gripper_block_dist, goal_block_dist, \
        #                        gripper_dist], \
        #                     axis=1)

        # return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
        #                        gripper_dist, \
        #                        gripper_block_z_dist, goal_block_z_dist, goal_gripper_z_dist], \
        #                     axis=1)
        # return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
        #                        obs[:, 3:5], \
        #                        gripper_block_z_dist, goal_block_z_dist, goal_gripper_z_dist], \
        #                     axis=1)
        # return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, obs[:, 3:5], obs[:, 0:3], obs[:, 5:8], obs[:, 8:11]], axis=1)

    def get_distance(self, obs):
        # gripper_pos = obs[:, 0:3]
        # block_pos = obs[:, 10:13]
        # goal_pos = obs[:, -6:-3]

        gripper_pos = obs[:, 0:3]
        block_pos = obs[:, 3:6]
        goal_pos = obs[:, -3:]

        gripper_block_dist = np.sqrt(np.sum((gripper_pos-block_pos)**2, axis=-1, keepdims=True))
        goal_block_dist = np.sqrt(np.sum((goal_pos-block_pos)**2, axis=-1, keepdims=True))
        gripper_goal_dist = np.sqrt(np.sum((gripper_pos-goal_pos)**2, axis=-1, keepdims=True))
        gripper_dist = np.expand_dims(obs[:, 9] + obs[:, 10], axis=1)

        return gripper_block_dist, goal_block_dist, gripper_goal_dist, gripper_dist
    
    def get_z_distance(self, obs):
        # gripper_pos = obs[:, 0:3]
        # block_pos = obs[:, 10:13]
        # goal_pos = obs[:, -6:-3]

        gripper_pos = obs[:, 0:3]
        block_pos = obs[:, 3:6]
        goal_pos = obs[:, -3:]

        return gripper_pos[:, -1:]-block_pos[:, -1:], goal_pos[:, -1:]-block_pos[:, -1:], goal_pos[:, -1:]-gripper_pos[:, -1:]

class AbsTransit_Push:
    def __init__(self):
        self.add_dim = 3

    def get_abs_obs(self, obs):
        # distance between objects
        gripper_block_dist, goal_block_dist, gripper_goal_dist, gripper_dist = self.get_distance(obs)
        # goal_block_angle, goal_gripper_angle, gripper_block_angle = self.get_angle(obs)
        gripper_angle, block_angle, goal_angle = self.get_angle_diff(obs)

        # return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
        #                        gripper_dist, \
        #                        goal_block_angle, goal_gripper_angle, gripper_block_angle], \
        #                     axis=1)
        return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
                               gripper_dist, \
                               gripper_angle, block_angle, goal_angle], \
                            axis=1)

    def get_distance(self, obs):
        # gripper_pos = obs[:, 0:3]
        # block_pos = obs[:, 10:13]
        # goal_pos = obs[:, -6:-3]
        gripper_pos = obs[:, 0:3]
        block_pos = obs[:, 3:6]
        goal_pos = obs[:, -3:]

        gripper_block_dist = np.sqrt(np.sum((gripper_pos-block_pos)**2, axis=-1, keepdims=True))
        goal_block_dist = np.sqrt(np.sum((goal_pos-block_pos)**2, axis=-1, keepdims=True))
        gripper_goal_dist = np.sqrt(np.sum((gripper_pos-goal_pos)**2, axis=-1, keepdims=True))
        gripper_dist = np.expand_dims(obs[:, 9] + obs[:, 10], axis=1)

        # only for debug
        # gripper_block_dist = 10 * gripper_block_dist
        # goal_block_dist = 10 * goal_block_dist
        # gripper_goal_dist = 10 * gripper_goal_dist

        return gripper_block_dist, goal_block_dist, gripper_goal_dist, gripper_dist
    
    def get_angle_diff(self, obs):
        # gripper_pos = obs[:, 0:3]
        # block_pos = obs[:, 10:13]
        # goal_pos = obs[:, -6:-3]
        gripper_pos = obs[:, 0:3]
        block_pos = obs[:, 3:6]
        goal_pos = obs[:, -3:]

        gripper_angle = np.arctan2(gripper_pos[:, 0] - block_pos[:, 0], gripper_pos[:, 1] - block_pos[:, 1]) -  \
                        np.arctan2(gripper_pos[:, 0] - goal_pos[:, 0], gripper_pos[:, 1] - goal_pos[:, 1])
        block_angle = np.arctan2(block_pos[:, 0] - gripper_pos[:, 0], block_pos[:, 1] - gripper_pos[:, 1]) -  \
                      np.arctan2(block_pos[:, 0] - goal_pos[:, 0], block_pos[:, 1] - goal_pos[:, 1])
        goal_angle = np.arctan2(goal_pos[:, 0] - block_pos[:, 0], goal_pos[:, 1] - block_pos[:, 1]) - \
                     np.arctan2(goal_pos[:, 0] - gripper_pos[:, 0], goal_pos[:, 1] - gripper_pos[:, 1])

        gripper_angle = np.abs((gripper_angle + np.pi) % (2*np.pi) - np.pi)
        block_angle = np.abs((block_angle + np.pi) % (2*np.pi) - np.pi)
        goal_angle = np.abs((goal_angle + np.pi) % (2*np.pi) - np.pi)

        # only for debug
        # gripper_angle = 0.1 * gripper_angle
        # block_angle = 0.1 * block_angle
        # goal_angle = 0.1 * goal_angle

        return np.expand_dims(gripper_angle, axis=1), np.expand_dims(block_angle, axis=1), np.expand_dims(goal_angle, axis=1)

class AbsTransit_Ant:
    def __init__(self):
        self.add_dim = 3

    def get_abs_obs(self, obs):
        # distance between objects
        x_dist, y_dist = self.get_distance(obs)

        return np.concatenate([x_dist, y_dist], axis=1)

    def get_distance(self, obs):
        ant_pos = obs[:, :2]
        goal_pos = obs[:, 29:31]

        return np.abs(ant_pos[:,:1]-goal_pos[:,:1]), np.abs(ant_pos[:,1:2]-goal_pos[:,1:2])

class AbsTransit_Block:
    def __init__(self):
        self.add_dim = 3
        self.block_num = None
    
    def set_num(self, block_num):
        self.block_num = block_num
        gripper_block_ids = np.arange(self.block_num)
        goal_block_dist = np.arange(self.block_num) + gripper_block_ids[-1] + 1
        gripper_goal_dist = np.arange(self.block_num) + goal_block_dist[-1] + 1
        gripper_dist = np.arange(self.block_num) + gripper_goal_dist[-1] + 1
        self.check_list = [gripper_block_ids, goal_block_dist, gripper_goal_dist, gripper_dist]

    def get_abs_obs(self, obs):
        # distance between objects
        gripper_block_dist, goal_block_dist, gripper_goal_dist, gripper_dist = self.get_distance(obs)

        return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
                               gripper_dist], \
                            axis=1)

    def get_distance(self, obs):
        # get block number
        block_num = (obs.shape[-1] - 13) // 18

        gripper_pos = obs[:, :3]
        block_pos_list = [obs[:, 10+i*15 : 10+i*15+3] for i in range(block_num)]
        goal_pos_list = [obs[:, 10+block_num*15+3*i : 10+block_num*15+3*i+3] for i in range(block_num)]

        gripper_block_dist_list = np.concatenate([np.sqrt(np.sum((gripper_pos-block_pos)**2, axis=-1, keepdims=True)) for block_pos in block_pos_list], axis=1)
        goal_block_dist_list = np.concatenate([np.sqrt(np.sum((goal_pos-block_pos)**2, axis=-1, keepdims=True)) for block_pos, goal_pos in zip(block_pos_list, goal_pos_list)], axis=1)
        gripper_goal_dist_list = np.concatenate([np.sqrt(np.sum((gripper_pos-goal_pos)**2, axis=-1, keepdims=True)) for goal_pos in goal_pos_list], axis=1)

        gripper_dist = np.expand_dims(obs[:, 3] + obs[:, 4], axis=1)

        return gripper_block_dist_list, goal_block_dist_list, gripper_goal_dist_list, gripper_dist

    # crop state dimension based on valid object id
    def do_crop_state(self, obs, obj_ids):
        # get block number
        block_num = (obs.shape[-1] - 13) // 18

        # comment state
        comment_state = obs[:, :10]
        # remain object state
        remain_obj_states = [obs[:, 10+i*15: 10+(i+1)*15] for i in obj_ids]
        # remain goal state
        remain_goal_states = [obs[:, 10+block_num*15+3*i:10+block_num*15+3*(i+1)] for i in obj_ids]
        # final state
        final_state = obs[:, -3:]

        return np.concatenate([comment_state]+remain_obj_states+remain_goal_states+[final_state], axis=1)

    # get obj id
    def get_obj_id(self, ori_id):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if ori_id <= each_list[-1]:
                return np.where(each_list == ori_id)[0][0]

        return None

    # abstract state id mapping if switch require
    def get_switch_ids(self, ori_id, goal_obj):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if ori_id <= each_list[-1]:
                return each_list[goal_obj]
            
        return None
    
    # check match
    def check_match_ids(self, ori_id, comp_id):
        # check
        assert self.block_num is not None

        # find
        for each_list in self.check_list:
            if (ori_id <= each_list[-1]) != (comp_id <= each_list[-1]):
                return False
            if ori_id <= each_list[-1]:
                return True

        return True