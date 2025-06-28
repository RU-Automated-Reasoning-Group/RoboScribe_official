import gym
import numpy as np
import math

import mujoco_py

import pdb

class GeneralEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # observation space
        self.env_obs_shape = self.observation_space['observation'].shape[0] + self.observation_space['desired_goal'].shape[0]
        self.observation_space = gym.spaces.Box(-math.inf, math.inf, shape=[self.env_obs_shape])

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def observation(self, obs):
        return np.concatenate([obs['observation'], obs['desired_goal']])
    
