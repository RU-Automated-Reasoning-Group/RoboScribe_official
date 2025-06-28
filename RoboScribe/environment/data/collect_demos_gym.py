
import argparse
import numpy as np
from environment.data.push_controller import get_push_control
from environment.data.pick_and_place_controller import get_pick_and_place_control
from environment.data.block_controller import get_block_control
from environment.utils.general_utils import AttrDict
from environment.general_env_gym import GeneralEnv
import gym
from tqdm import tqdm
import random
import os
import pickle
from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt

import pdb

class CollectDemos():
    """
    Class to generate a dataset of demonstrations for the Fetch environment tasks using a set of handcrafted controllers.

    """
    def __init__(self, demo_path, num_trajectories=5, subseq_len=10, task="block", env=None, traj_len=100, img_path=None, env_name=None, block_num=2):

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
        if env is not None:
            self.env = env
        if self.task == "push":
            self.env = GeneralEnv(gym.make('FetchPush-v2', max_episode_steps=100, render_mode='rgb_array'))
        elif self.task == 'pick':
            self.env = GeneralEnv(gym.make('FetchPickAndPlace-v2', max_episode_steps=100, render_mode='rgb_array'))
        elif self.task == 'block':
            assert env_name is not None
            self.env = GeneralEnv(gym.make(env_name, stack_only=True))
            self.block_limit = 0
            self.block_num = block_num

    def get_p_noise(self, idx, factor):
        a = np.array([self.x_noise(idx/factor), self.y_noise(idx/factor), self.z_noise(idx/factor), 0])
        return a

    def get_obs(self, obs):
        return np.concatenate([obs['observation'], obs['desired_goal']])

    def collect(self, store=True):
        print("Collecting demonstrations...")
        obs_imgs = []

        collect_traj_num = 0

        # for i in tqdm(range(self.num_trajectories)):
        while collect_traj_num < self.num_trajectories:
            if self.task == 'block':
                self.block_limit = 0
            if collect_traj_num % 10 == 0:
                print('current collect num: {}'.format(collect_traj_num))
            obs_imgs.append([])

            pdb.set_trace()

            obs = self.env.reset()
            done = False
            actions = []
            observations = []
            terminals = []

            self.x_noise = PerlinNoise(octaves=3)
            self.y_noise = PerlinNoise(octaves=3)
            self.z_noise = PerlinNoise(octaves=3)

            if self.task == 'push':
                controller = get_push_control
            elif self.task == 'pick':
                controller = get_pick_and_place_control
            elif self.task == 'block':
                controller = get_block_control
            else:
                pdb.set_trace()

            idx = 0

            while not done:

                print(self.block_limit)

                observations.append(obs)

                p_noise = self.get_p_noise(idx, 1000)
                idx += 1

                if self.task == 'block':
                    action, success, cur_block_id = controller(obs, self.block_limit)
                    if action is None:
                        print('wrong block id, current block id: {}'.format(cur_block_id))
                        break
                    else:
                        if success:
                            self.block_limit += 1
                            if cur_block_id != self.block_num-1:
                                success = False
                        else:
                            self.block_limit = cur_block_id
                else:
                    action, success = controller(obs)
    
                action += p_noise * 0.5
                actions.append(action)

                obs, _, done, _ = self.env.step(action)
                terminals.append(success)

                img_array = self.env.render(mode='rgb_array')
                obs_imgs[-1].append(img_array)

                if success:
                    break
                elif len(actions) >= self.traj_len:
                    break

            # only for debug
            plt.figure()
            for im_id, img in enumerate(obs_imgs[-1]):
                plt.imshow(img)
                plt.title(im_id)
                plt.savefig('test_collect/{}.png'.format(im_id))
                plt.cla()
            plt.close()

            if len(actions) <= self.subseq_len+1:
                obs_imgs.pop(-1)
                continue
            elif not success:
                obs_imgs.pop(-1)
                continue
            else:
                collect_traj_num += 1
                self.seqs.append(AttrDict(
                    obs=observations,
                    actions=actions,
                    ))


        np_seq = np.array(self.seqs)
        if store and self.save_dir is not None:
            np.save(self.save_dir, np_seq)
            if self.img_path is not None:
                with open(self.img_path, 'wb') as f:
                    pickle.dump(obs_imgs, f)

        print("Dataset Generated.")

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