import mani_skill2.envs
from mani_skill2.utils.geometry import transform_points
import gymnasium
import gym
from gym.spaces import Box

import numpy as np

import pdb

class OpendrawerObswrap(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = Box(shape=(80,), low=-np.inf, high=np.inf)

    def reset(self, **kwargs):
        return self.observation(self.env.reset(**kwargs))
    
    def observation(self, obs):
        add_1 = self.env.link_qpos
        add_2 = self.env.target_qpos
        add_3 = transform_points(self.env.target_link.pose.to_transformation_matrix(), self.env.target_handle_pcd).mean(axis=0)
        obs = np.concatenate([obs, np.array([add_1]), np.array([add_2]), add_3], axis=0)

        return obs
    
class OpendrawerPickPlaceCubeObswrap(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(shape=(115,), low=-np.inf, high=np.inf)

    def reset(self, **kwargs):
        return self.observation(self.env.reset(**kwargs))
    
    def observation(self, obs):
        add_1 = self.env.link_qpos
        add_2 = self.env.target_qpos
        add_3 = transform_points(self.env.target_link.pose.to_transformation_matrix(), self.env.target_handle_pcd).mean(axis=0)
        obs = np.concatenate([obs, np.array([add_1]), np.array([add_2]), add_3], axis=0)

        return obs
    
    def get_full_obs(self):
        obs = self.env.get_full_obs()
        add_1 = self.env.link_qpos
        add_2 = self.env.target_qpos
        add_3 = transform_points(self.env.target_link.pose.to_transformation_matrix(), self.env.target_handle_pcd).mean(axis=0)
        obs = np.concatenate([obs, np.array([add_1]), np.array([add_2]), add_3], axis=0)

        return obs

class OpendrawerRewwrap(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        obs = self.get_obs()
        info = self.get_info(obs=obs)
        if info['success']:
            return 0.
        else:
            return -1.

class PickPlaceCubeRewwrap(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        obs = self.get_obs()
        success = False
        if self.solve_cube==1 and \
            self.check_cube_in_drawer(self.cubeA):
            success = True
        elif self.solve_cube==2 and \
            self.check_cube_in_drawer(self.cubeA) and self.check_cube_in_drawer(self.cubeB):
            success = True
        elif self.solve_cube==3 and \
            self.check_cube_in_drawer(self.cubeA) and self.check_cube_in_drawer(self.cubeB) and self.check_cube_in_drawer(self.cubeC):
            success = True
        # if self.check_cube_in_drawer(self.cubeA):
        #     success = True

        if success:
            return 0.
        else:
            return -1.

class PickPlaceCubeRewwrapDebug(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(shape=(14,), low=-np.inf, high=np.inf)

    def reward(self, reward):
        obs = self.get_obs()
        success = False
        if self.solve_cube==1 and \
            self.check_cube_in_drawer(self.cubeA):
            success = True
        elif self.solve_cube==2 and \
            self.check_cube_in_drawer(self.cubeA) and self.check_cube_in_drawer(self.cubeB):
            success = True
        elif self.solve_cube==3 and \
            self.check_cube_in_drawer(self.cubeA) and self.check_cube_in_drawer(self.cubeB) and self.check_cube_in_drawer(self.cubeC):
            success = True
        # if self.check_cube_in_drawer(self.cubeA):
        #     success = True

        if success:
            return 0.
        else:
            return -1.

def opendrawer_dropobs(obs):
    return obs[:-5]

def get_opendrawer():
    env = gym.make('OpenCabinetDrawer-v1', obs_mode="state", reward_mode='dense', model_ids=['1000'], fixed_target_link_idx=1, \
                   asset_root='environment/ManiSkill2-0.4.2/data/partnet_mobility/dataset')
    env = OpendrawerRewwrap(OpendrawerObswrap(env))
    return env

def get_opendrawer_v3():
    env = gym.make("OpenCabinetDrawerCube2ArmToppanel-v3", task="opendrawer", extract="drawer", obs_mode="state", reward_mode="dense", \
                   asset_root='environment/ManiSkill2-0.4.2/data/partnet_mobility/dataset', model_ids=['999'], fixed_target_link_idx=1)
    env.set_extract("drawer")
    env = OpendrawerRewwrap(OpendrawerObswrap(env))
    # env = OpendrawerRewwrap(env)

    return env


def get_pick_place_cube():
    env = gym.make("OpenCabinetDrawerCube2ArmToppanel-v2", task="pickplace", extract="cube", obs_mode="state", reward_mode="dense", \
                   asset_root='environment/ManiSkill2-0.4.2/data/partnet_mobility/dataset', model_ids=['999'], fixed_target_link_idx=1)
    env.set_extract("cube")
    env = PickPlaceCubeRewwrap(env)

    return env

def get_pick_place_cube_debug(solve_cube=2):
    env = gym.make("OpenCabinetDrawerCube2ArmToppanel-v3", task="pickplace", extract="cube", obs_mode="state_dict", reward_mode="dense", \
                   asset_root='environment/ManiSkill2-0.4.2/data/partnet_mobility/dataset', model_ids=['999'], fixed_target_link_idx=1, \
                   solve_cube=solve_cube)
    env.set_extract("cube")
    env = PickPlaceCubeRewwrap(env)

    return env

def get_pick_place_cube_multi(solve_cube=2, reassign=False):
    env = gym.make("OpenCabinetDrawerCube2ArmToppanel-v3", task="pickplace", extract="cube", obs_mode="state", reward_mode="dense", \
                   asset_root='environment/ManiSkill2-0.4.2/data/partnet_mobility/dataset', model_ids=['999'], fixed_target_link_idx=1, \
                   solve_cube=solve_cube, reassign=reassign)
    env.set_extract("cube")
    env = PickPlaceCubeRewwrap(env)

    return env

def get_drawer_pick_place_cube_multi(solve_cube=2, reassign=False):
    env = gym.make("OpenCabinetDrawerCube2ArmToppanel-v3", task="drawer_cube", extract="cube", obs_mode="state", reward_mode="dense", \
                   asset_root='environment/ManiSkill2-0.4.2/data/partnet_mobility/dataset', model_ids=['999'], fixed_target_link_idx=1, \
                   solve_cube=solve_cube, reassign=reassign)
    env.set_extract("drawer_cube")
    env = PickPlaceCubeRewwrap(OpendrawerPickPlaceCubeObswrap(env))
    # env = PickPlaceCubeRewwrapDebug(OpendrawerPickPlaceCubeObswrap(env))

    return env

def get_drawer_pick_place_cube_multi_debug(solve_cube=2, reassign=False):
    env = gym.make("OpenCabinetDrawerCube2ArmToppanel-v3", task="opendrawer", extract="cube", obs_mode="state", reward_mode="dense", \
                   asset_root='environment/ManiSkill2-0.4.2/data/partnet_mobility/dataset', model_ids=['999'], fixed_target_link_idx=1, \
                   solve_cube=solve_cube, reassign=reassign)
    env.set_extract("drawer_cube")
    env = PickPlaceCubeRewwrap(OpendrawerPickPlaceCubeObswrap(env))

    return env