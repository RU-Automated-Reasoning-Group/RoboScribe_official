import gymnasium as gym
from gymnasium.spaces import Box, Dict
import gym.spaces as gym_space
import numpy as np
import math

import mujoco_py

import pdb

class GymToGymnasium(gym.Wrapper):
    def __init__(self, env, env_success_rew=0, new_observation_space=None, new_action_space=None, render_mode='rgb_array'):
        super().__init__(env)
        # init
        self.env_success_rew = env_success_rew

        # observation space
        if new_observation_space is None:
            ori_obs_space = self.env.observation_space
            if isinstance(ori_obs_space, gym_space.Dict):
                obs_space = {k:Box(low=ori_obs_space[k].low, \
                            high=ori_obs_space[k].high, \
                            shape=ori_obs_space[k].shape, \
                            dtype=ori_obs_space[k].dtype) 
                            for k in ori_obs_space.spaces}
                self.observation_space = Dict(obs_space)
            elif isinstance(ori_obs_space, gym_space.Box):
                obs_space = Box(low=ori_obs_space.low, high=ori_obs_space.high, shape=ori_obs_space.shape, dtype=ori_obs_space.dtype)
                self.observation_space = obs_space
        else:
            self.observation_space = new_observation_space
        
        # action space
        if new_action_space is None:
            ori_act_space = self.env.action_space
            self.action_space = Box(low=ori_act_space.low, \
                                high=ori_act_space.high, \
                                shape=ori_act_space.shape, \
                                dtype=ori_act_space.dtype)
        else:
            self.action_space = new_action_space

        # others
        self.user_render_mode = render_mode

    @property
    def render_mode(self):
        return self.user_render_mode
    
    @render_mode.setter
    def render_mode(self, new_mode):
        self.user_render_mode = new_mode

    def reset(self, goal_id=None, **kwargs):
        obs = self.env.reset()
        return obs, {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = done
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render(mode=self.user_render_mode)

class GeneralEnv(gym.ObservationWrapper):
    def __init__(self, env, env_success_rew=0, obs_key='observation', goal_key='desired_goal'):
        super().__init__(env)
        # observation space
        self.env_obs_shape = self.observation_space[obs_key].shape[0] + self.observation_space[goal_key].shape[0]
        self.observation_space = gym.spaces.Box(-math.inf, math.inf, shape=[self.env_obs_shape])
        self.env_success_rew = env_success_rew
        # others
        self.obs_key = obs_key
        self.goal_key = goal_key

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        return np.concatenate([obs[self.obs_key], obs[self.goal_key]])
    
class GeneralDebugEnv(gym.ObservationWrapper):
    def __init__(self, env, env_success_rew=0):
        super().__init__(env)
        # observation space
        self.env_obs_shape = 28
        self.observation_space = gym.spaces.Box(-math.inf, math.inf, shape=[self.env_obs_shape])
        self.env_success_rew = env_success_rew

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        obs = self.env.get_custom_obs(0)
        return obs
    
