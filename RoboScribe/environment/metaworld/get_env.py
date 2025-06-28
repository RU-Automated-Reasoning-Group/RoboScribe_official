import mani_skill2.envs
from mani_skill2.utils.geometry import transform_points
import gymnasium
import gym
import lorl_env
from gym.spaces import Box

import numpy as np

import pdb


class MetaworldObswrap(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = Box(shape=(22,), low=-np.inf, high=np.inf)
        self.observation_space = Box(shape=(18,), low=-np.inf, high=np.inf)

    def reset(self, **kwargs):
        return self.observation(self.env.reset(**kwargs))
    
    def observation(self, obs):
        add_obs = np.concatenate([obs['initial_state'][4:7], obs['initial_state'][13:14]])
        obs = np.concatenate([obs['current_state'], add_obs])

        return obs
    

class MetaworldRewwrap(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        cur_obs = self.env.get_obs()
        success = self.env.success_white_mug_back(cur_obs) and \
                  self.env.success_faucet_left(cur_obs) and \
                  self.env.success_drawer_open(cur_obs)

        if success:
            return 0.
        else:
            return -1.

def metaworld_dropobs(obs):
    return obs[:-4]

def get_metaworld():
    env = gym.make("Lorl-v2")
    env = MetaworldRewwrap(MetaworldObswrap(env))
    return env
