import gymnasium as gym
import numpy as np
import os
import math

import matplotlib.pyplot as plt

from modules.reward import RewIdentity, RewDebug
from utils.logging import init_logging, log_and_print

import pdb

class SkillEnv(gym.Wrapper):
    def __init__(self, env, skill_list, rew_fun, threshold=1.0, traj_len=100, hold_len=1, train=False, lagrangian=False, traj_limit_mode=False):
        super().__init__(env)
        self.attemp_num = 200
        self.skill_list = skill_list
        self.rew_fun = rew_fun
        self.done_thre = threshold
        self.traj_len = traj_len
        self.traj_id = 0
        self.train = train

        # should be no less than 1
        self.hold_len = hold_len
        self.hold_step = 0
        assert self.hold_len >= 1

        self.debug_id = 0
        self.reset_time = 0
        self.lagrangian = lagrangian
        self.traj_limit_mode = traj_limit_mode

        # only for debug
        # self.env_obs_shape = self.observation_space.shape[0] + 2
        # self.observation_space = gym.spaces.Box(-math.inf, math.inf, shape=[self.env_obs_shape])


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
        while True:
            # reset
            obs, info = self.env.reset(**kwargs)

            terminated = truncated = False
            self.traj_id = 0

            # apply skills
            success = len(self.skill_list) == 0
            for skill in self.skill_list:
                skill.reset()
                # debug
                if self.traj_limit_mode:
                    traj_len_limit = min(self.traj_len-self.traj_id, skill.traj_len_limit)
                else:
                    traj_len_limit = self.traj_len-self.traj_id
                obs, reward, terminated, truncated, info, traj_len = skill.rollout(self.env, obs, traj_len_limit, drop_success=False)

                self.traj_id += traj_len
                if self.traj_limit_mode:
                    success = self.traj_len > self.traj_id
                else:
                    success = self.traj_len > self.traj_id and terminated
                if not success:
                    break

            # success
            if success:
                break
            else:
                cur_attempt += 1
                if cur_attempt > self.attemp_num:
                    raise Exception('reset fail on skill environment')

        # only for debug
        # obs = np.concatenate([obs, np.array([0, 0])])
        self.reset_time += 1

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.lagrangian and not isinstance(self.rew_fun, RewIdentity) and self.train:
            new_reward, new_cost = self.rew_fun.get_reward(obs, reward)
            info['cost'] = new_cost
        else:
            new_reward = self.rew_fun.get_reward(obs, reward)
            info['cost'] = 0
        self.traj_id += 1

        # train mode
        if self.train:
            truncated = self.traj_id >= self.traj_len
            # penalty
            if not isinstance(self.rew_fun, RewIdentity):
                # new_reward -= 1
                # only for debug
                new_reward -= 0
                if self.lagrangian:
                    if new_reward >= 0 and new_cost <= 0:
                        new_reward += 1
                elif new_reward >= 0:
                    new_reward += 1
            else:
                if new_reward == -1:
                    new_reward = -1
                elif new_reward == 0:
                    new_reward = 1
                else:
                    pdb.set_trace()
                    raise NotImplementedError
            # new_reward -= 0.1 * np.linalg.norm(action)

            # only for debug
            # rew_details = self.rew_fun.equ.execute_dense_details(np.expand_dims(obs, axis=0))
            # add_obs = (np.array(rew_details[0]) > 0).astype(float)
            # obs = np.concatenate([obs, add_obs])

            return obs, new_reward, False, truncated, info

        # eval mode
        else:
            # environment success
            if reward >= 0:
                truncated = self.traj_id >= self.traj_len
                terminated = True
                new_reward = 1
            else:
                if new_reward >= self.done_thre:
                    self.hold_step += 1
                else:
                    # attempt to set negative (debug)
                    # if not isinstance(self.rew_fun, RewIdentity) and not isinstance(self.rew_fun, RewDebug):
                        # new_reward -= self.done_thre
                        # new_reward -= 1
                    self.hold_step = 0
                truncated = self.traj_id >= self.traj_len
                terminated = self.hold_step >= self.hold_len

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

        # return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
        #                        gripper_dist], \
        #                     axis=1)

        return np.concatenate([gripper_block_dist, goal_block_dist, gripper_goal_dist, \
                               gripper_dist, \
                               gripper_block_z_dist, goal_block_z_dist, goal_gripper_z_dist], \
                            axis=1)
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
