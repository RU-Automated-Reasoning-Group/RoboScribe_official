
import argparse
import numpy as np
from environment.data.push_controller import get_push_control
from environment.data.pick_and_place_controller import get_pick_and_place_control
from environment.data.block_controller import get_block_control
from environment.data.custom_block_controller import get_custom_block_control
from environment.data.pickplacemulti_controller import get_pickmulti_control
from environment.utils.general_utils import AttrDict
from environment.general_env import GeneralEnv, GymToGymnasium
from environment.fetch_custom.get_fetch_env import get_env, get_pickplace_env, episode_render_fn
from environment.cee_us_env.fpp_construction_env import FetchPickAndPlaceConstruction
# import gym
import gymnasium as gym
from tqdm import tqdm
import random
import os
import copy
import pickle
from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt

import pdb

class CollectDemos():
    """
    Class to generate a dataset of demonstrations for the Fetch environment tasks using a set of handcrafted controllers.

    """
    def __init__(self, demo_path, num_trajectories=5, subseq_len=10, task="block", env=None, traj_len=100, img_path=None, env_name=None, block_num=3, debug=False):

        self.seqs = []
        self.task = task
        # self.dataset_dir = "../dataset/" + dataset_name + "/"
        # os.makedirs(self.dataset_dir, exist_ok=True)
        # self.save_dir = "../dataset/" + dataset_name + "/" + "demos.npy"
        self.save_dir = demo_path
        self.img_path = img_path
        self.num_trajectories = num_trajectories
        self.subseq_len = subseq_len
        self.traj_len = traj_len
        self.debug = debug
        if env is not None:
            self.env = env
        if self.task == "push":
            self.env = GeneralEnv(gym.make('FetchPush-v2', max_episode_steps=100, render_mode='rgb_array'))
        elif self.task == 'pick':
            self.env = GeneralEnv(gym.make('FetchPickAndPlace-v2', max_episode_steps=100, render_mode='rgb_array'))
        elif self.task == 'pickmulti':
            self.env = GymToGymnasium(FetchPickAndPlaceConstruction(name=env_name, sparse=False, shaped_reward=False, num_blocks=block_num, reward_type='sparse', case = 'PickAndPlace', visualize_mocap=False, simple=True))
            self.block_num = block_num
        elif self.task == 'block':
            assert env_name is not None
            self.env = GeneralEnv(gym.make(env_name, stack_only=True, max_episode_steps=200, render_mode='rgb_array'))
            self.block_id = 0
            self.block_num = block_num
        elif self.task == 'custom_block':
            assert env_name is not None
            self.env = GeneralEnv(get_env(timesteps=traj_len, eval=False, block_num=block_num), env_success_rew=0, goal_key='goal')
            self.extra_env = get_env(timesteps=traj_len, eval=False, block_num=block_num)
            self.block_id = block_num-1
            self.block_num = block_num
        elif self.task == 'custom_pick':
            assert env_name is not None
            self.env = GeneralEnv(get_pickplace_env(timesteps=traj_len, eval=False), env_success_rew=0, goal_key='goal')
            self.extra_env = get_pickplace_env(timesteps=traj_len, eval=False)
            self.block_id = 0
            self.block_num = 1

    def get_p_noise(self, idx, factor):
        a = np.array([self.x_noise(idx/factor), self.y_noise(idx/factor), self.z_noise(idx/factor), 0])
        return a

    def get_obs(self, obs):
        return np.concatenate([obs['observation'], obs['desired_goal']])

    def collect(self, store=True, seeds=None, require_id=False):
        print("Collecting demonstrations...")
        obs_imgs = []
        # data_dict = {'obs':[], 'imgs':[], 'act':[], 'seed':[]}
        # seeds = np.arange(10)

        debug_list = []
        collect_traj_num = 0

        # for i in tqdm(range(self.num_trajectories)):
        while collect_traj_num < self.num_trajectories:
            if self.task == 'block' or self.task == 'pickmulti':
                self.block_id = 0

            if collect_traj_num % 1 == 0:
                print('current collect num: {}'.format(collect_traj_num))
            obs_imgs.append([])

            if seeds is None:
                obs, _ = self.env.reset()
            else:
                obs, _ = self.env.reset(seed=int(seeds[collect_traj_num]))
            if self.task == 'custom_block':
                # find order of blocks
                all_blocks = [dims[-1] for dims in np.split(obs[19:], 3)]
                all_blocks_idx = np.argsort(all_blocks)
                self.block_id = 0
                debug_list.append(self.env.env.env._env._env._env.goal_idx)
            elif self.task == 'custom_pick':
                all_blocks_idx = [0]
                self.block_id = 0
                debug_list.append(self.env.env.env._env._env._env.goal_idx)

            done = False
            actions = []
            observations = []
            terminals = []
            if require_id:
                obj_ids = []

            self.x_noise = PerlinNoise(octaves=3)
            self.y_noise = PerlinNoise(octaves=3)
            self.z_noise = PerlinNoise(octaves=3)

            if self.task == 'push':
                controller = get_push_control
            elif self.task == 'pick':
                controller = get_pick_and_place_control
            elif self.task == 'block':
                controller = get_block_control
            elif self.task == 'custom_block' or self.task == 'custom_pick':
                controller = get_custom_block_control
            elif self.task == 'pickmulti':
                controller = get_pickmulti_control
            else:
                pdb.set_trace()

            idx = 0

            # only for debug
            # plt.figure()
            while not done:
                if self.debug:
                    observations.append(self.env.env.get_custom_obs(0))
                else:
                    observations.append(obs)
                if require_id:
                    obj_ids.append(self.block_id)

                img_array = self.env.render()
                obs_imgs[-1].append(img_array)

                p_noise = self.get_p_noise(idx, 1000)
                idx += 1

                if self.task == 'block' or self.task == 'pickmulti':
                    action, success = controller(obs, block_id=self.block_id, last_block=self.block_id==(self.block_num-1))
                    if action is None:
                        print('wrong block id')
                        break
                    else:
                        if success:
                            if self.block_id < self.block_num - 1:
                                self.block_id += 1
                                success = False
                elif self.task == 'custom_block' or self.task == 'custom_pick':
                    new_obs = copy.deepcopy(obs)
                    new_obs[:obs.shape[0]//2] = self.env.obs_min + new_obs[:obs.shape[0]//2] * (self.env.obs_max -  self.env.obs_min)
                    new_obs[obs.shape[0]//2:] = self.env.obs_min + new_obs[obs.shape[0]//2:] * (self.env.obs_max -  self.env.obs_min)
                    action, success = controller(new_obs, all_blocks_idx, block_id=all_blocks_idx[self.block_id], last_block=self.block_id==self.block_num-1)
                    if action is None:
                        print('wrong block id')
                        break
                    else:
                        if success:
                            if self.block_id < self.block_num - 1:
                                self.block_id += 1
                                success = False
                else:
                    action, success = controller(obs)
    
                action += p_noise * 0.5
                actions.append(action)

                obs, reward, done, _, _ = self.env.step(action)
                terminals.append(success)

                if success or (self.task=='block' and reward==1) or (self.task=='custom_block' and reward==0) or (self.task=='custom_pick' and reward==0):
                    success = True
                    break
                elif len(actions) >= self.traj_len:
                    break

                # only for debug
                # plt.imshow(img_array)
                # plt.title(idx)
                # plt.savefig('test_collect/debug/{}.png'.format(idx))
                # plt.cla()

            # get the last one
            if self.debug:
                observations.append(self.env.env.get_custom_obs(0))
            else:
                observations.append(obs)
            if require_id:
                obj_ids.append(self.block_id)
            obs_imgs[-1].append(self.env.render())

            if len(actions) <= self.subseq_len+1:
                obs_imgs.pop(-1)
                continue
            elif not success:
                # for im_id, img_array in enumerate(obs_imgs[-1]):
                #     plt.imshow(img_array)
                #     plt.title(im_id)
                #     plt.savefig('test_collect/debug/{}.png'.format(im_id))
                #     plt.cla()
                obs_imgs.pop(-1)
                continue
            elif self.task == 'pickmulti' and reward < 0:
                # for im_id, img_array in enumerate(obs_imgs[-1]):
                #     plt.imshow(img_array)
                #     plt.title(im_id)
                #     plt.savefig('test_collect/debug/{}.png'.format(im_id))
                #     plt.cla()
                obs_imgs.pop(-1)
                continue
            else:
                collect_traj_num += 1
                if require_id:
                    self.seqs.append(AttrDict(
                        obs=observations,
                        actions=actions,
                        obj_ids=obj_ids
                        ))
                else:
                    self.seqs.append(AttrDict(
                        obs=observations,
                        actions=actions,
                        ))
                # data_dict['obs'].append(observations)
                # data_dict['act'].append(actions)
                # data_dict['seed'].append(int(seeds[collect_traj_num-1]))
                # data_dict['imgs'].append(obs_imgs[-1])
            # only for debug
            # plt.close()

        np_seq = np.array(self.seqs)
        if store and self.save_dir is not None:
            np.save(self.save_dir, np_seq)
            # np.savez(self.save_dir, **data_dict)
            if self.img_path is not None:
                with open(self.img_path, 'wb') as f:
                    pickle.dump(obs_imgs, f)

        print("Dataset Generated.")
        # print(set(debug_list))
        # print(debug_list)

        return np_seq, obs_imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trajectories', type=int, default=10)
    parser.add_argument('--subseq_len', type=int, default=10)
    parser.add_argument('--task', type=str, default="block", choices=["block", "hook", "pick"])
    args = parser.parse_args()

    dataset_name = "fetch_" + args.task + "_" + str(args.num_trajectories)
    collector = CollectDemos(dataset_name=dataset_name, num_trajectories=args.num_trajectories, subseq_len=args.subseq_len, task=args.task)
    collect_obs, collect_imgs = collector.collect()

    plt.figure()
    for img_id, img in enumerate(collect_imgs):
        # get observation
        cur_obs = collect_obs[img_id]
        extra_str = str(img_id) + '\n' + \
                    ',  '.join([str(round(element.item(), 3)) for element in cur_obs[:3]]) + '  |  ' + \
                    ',  '.join([str(round(element.item(), 3)) for element in cur_obs[3:5]]) + '\n' + \
                    ',  '.join([str(round(element.item(), 3)) for element in cur_obs[10:13]]) + '  |  ' + \
                    ',  '.join([str(round(element.item(), 3)) for element in cur_obs[-6:-3]])
        # store image
        plt.imshow(img)
        plt.title(extra_str)
        plt.savefig(os.path.join(dataset_name, '{}.png'.format(img_id)))
        plt.cla()