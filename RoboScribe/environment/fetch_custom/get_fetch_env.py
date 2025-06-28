from environment.fetch_custom.customfetch.custom_fetch import WallsDemoStackEnv, PickPlaceEnv, WallsDemoSingleStackEnv
from environment.fetch_custom.customfetch_wrapper import ConvertGoalEnvWrapper, NormObsWrapper, ClipObsWrapper, GymWrapper, OneHotAction, NormalizeAction
# from customfetch.custom_fetch import WallsDemoStackEnv, PickPlaceEnv, WallsDemoSingleStackEnv
# from customfetch_wrapper import ConvertGoalEnvWrapper, NormObsWrapper, ClipObsWrapper, GymWrapper, OneHotAction, NormalizeAction
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from gymnasium.wrappers import StepAPICompatibility

import matplotlib.pyplot as plt

import pdb

class GymToGymnasium(gym.Wrapper):
    def __init__(self, env, render_mode='rgb_array'):
        super().__init__(env)
        # reset observation space and action space
        ori_obs_space = self.env.observation_space
        ori_act_space = self.env.action_space
        obs_space = {k:Box(low=ori_obs_space[k].low, \
                          high=ori_obs_space[k].high, \
                          shape=ori_obs_space[k].shape, \
                          dtype=ori_obs_space[k].dtype) 
                        for k in ori_obs_space.spaces}
        self.observation_space = Dict(obs_space)
        self.action_space = Box(low=ori_act_space.low, \
                                high=ori_act_space.high, \
                                shape=ori_act_space.shape, \
                                dtype=ori_act_space.dtype)
        # others
        self.test = 'ha'
        self.user_render_mode = render_mode

        # only available for block stack
        self.candidate_ids = [9,10,11,12,13,14]

    @property
    def render_mode(self):
        return self.user_render_mode
    
    @render_mode.setter
    def render_mode(self, new_mode):
        self.user_render_mode = new_mode

    def reset(self, goal_id=None, **kwargs):
        # randomly set goal id
        # if goal_id is None:
        #     goal_id = np.random.choice(self.candidate_ids)
        # self.env._env._env._env.set_goal_idx(goal_id)
        if goal_id is not None:
           self.env._env._env._env.set_goal_idx(goal_id)

        obs = self.env.reset()
        return obs, {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = self.env.num_step >= self.env.max_step
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render(mode=self.user_render_mode)

class GymToGymnasiumSingle(gym.Wrapper):
    def __init__(self, env, render_mode='rgb_array'):
        super().__init__(env)
        # reset observation space and action space
        ori_obs_space = self.env.observation_space
        ori_act_space = self.env.action_space
        obs_space = {k:Box(low=ori_obs_space[k].low, \
                          high=ori_obs_space[k].high, \
                          shape=ori_obs_space[k].shape, \
                          dtype=ori_obs_space[k].dtype) 
                        for k in ori_obs_space.spaces}
        self.observation_space = Dict(obs_space)
        self.action_space = Box(low=ori_act_space.low, \
                                high=ori_act_space.high, \
                                shape=ori_act_space.shape, \
                                dtype=ori_act_space.dtype)
        # others
        self.test = 'ha'
        self.user_render_mode = render_mode

        # only available for block stack
        self.candidate_ids = [0,1,2,3,4,5]
        self.init_ids = [0,1,2]

    @property
    def render_mode(self):
        return self.user_render_mode
    
    @render_mode.setter
    def render_mode(self, new_mode):
        self.user_render_mode = new_mode

    def reset(self, **kwargs):
        # randomly set initial id
        init_id = np.random.choice(self.init_ids)
        self.env._env._env._env.set_init_pos(init_id)

        # randomly set goal id
        goal_id = np.random.choice(self.candidate_ids)
        self.env._env._env._env.set_goal_idx(goal_id)

        obs = self.env.reset()
        return obs, {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = self.env.num_step >= self.env.max_step
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render(mode=self.user_render_mode)


def wrap_mega_env(e, info_to_obs_fn=None):
    e = GymWrapper(e, info_to_obs_fn=info_to_obs_fn)
    pdb.set_trace()
    if hasattr(e.act_space['action'], 'n'):
        e = OneHotAction(e)
    else:
        e = NormalizeAction(e)
    return e

# get environment
def get_env(timesteps, eval, block_num):
    env = WallsDemoStackEnv(max_step=timesteps, eval=eval, n=block_num)

    # Antmaze is a GoalEnv
    env = ConvertGoalEnvWrapper(env)
    # LEXA assumes information is in obs dict already, so move info dict into obs.
    info_to_obs = None
    def info_to_obs(info, obs):
        if info is None:
          info = env.get_metrics_dict()
        obs = obs.copy()
        for k,v in info.items():
          if eval:
            if "metric" in k:
              obs[k] = v
          else:
            if "above" in k:
              obs[k] = v
        return obs
        
    obs_min = np.ones(env.observation_space['observation'].shape) * -1e6
    pos_min = [1.0, 0.3, 0.35]

    # demofetchpnp in task
    obs_min[:3] = obs_min[5:8] = obs_min[8:11] = pos_min
    if env.n == 3:
        obs_min[11:14] = pos_min

    obs_max = np.ones(env.observation_space['observation'].shape) * 1e6
    pos_max = [1.6, 1.2, 1.0]

    # demofetchpnp in task
    obs_max[:3] = obs_max[5:8] = obs_max[8:11] = pos_max
    if env.n == 3:
        obs_max[11:14] = pos_max

    env = ClipObsWrapper(env, obs_min, obs_max)

    # first 3 dim are grip pos, next 2 dim are gripper, next n * 3 are obj pos.
    # if block_num == 2: # noisy dim
    #   obs_min_noise = np.ones(noise_dim) * noise_low
    #   obs_min = np.concatenate([env.workspace_min, [0., 0.],  *[env.workspace_min for _ in range(env.n)], obs_min_noise], 0)
    #   obs_max_noise = np.ones(noise_dim) * noise_high
    #   obs_max = np.concatenate([env.workspace_max, [0.05, 0.05],  *[env.workspace_max for _ in range(env.n)], obs_max_noise], 0)
    # else:
    obs_min = np.concatenate([env.workspace_min, [0., 0.],  *[env.workspace_min for _ in range(env.n)]], 0)
    obs_max = np.concatenate([env.workspace_max, [0.05, 0.05],  *[env.workspace_max for _ in range(env.n)]], 0)
    env = NormObsWrapper(env, obs_min, obs_max)
    # env = wrap_mega_env(env, info_to_obs)

    # obs = env.reset()
    # obs, _, _, _ = env.step(np.array([0,0,0,0]))
    # pdb.set_trace()
    # episode_render_fn(env, {'goal':[obs['goal']], 'observation':[obs['observation']]})
    # pdb.set_trace()
    env = GymToGymnasium(env)
    
    return env

# get pick and place environment
def get_pickplace_env(timesteps, eval):
    env = WallsDemoSingleStackEnv(max_step=timesteps, eval=False)

    # Antmaze is a GoalEnv
    env = ConvertGoalEnvWrapper(env)
    # LEXA assumes information is in obs dict already, so move info dict into obs.
    info_to_obs = None
    def info_to_obs(info, obs):
        if info is None:
          info = env.get_metrics_dict()
        obs = obs.copy()
        for k,v in info.items():
          if eval:
            if "metric" in k:
              obs[k] = v
          else:
            if "above" in k:
              obs[k] = v
        return obs
        
    obs_min = np.ones(env.observation_space['observation'].shape) * -1e6
    pos_min = [1.0, 0.3, 0.35]

    # demofetchpnp in task
    obs_min[:3] = obs_min[5:8] = pos_min

    obs_max = np.ones(env.observation_space['observation'].shape) * 1e6
    pos_max = [1.6, 1.2, 1.0]

    # demofetchpnp in task
    obs_max[:3] = pos_max

    env = ClipObsWrapper(env, obs_min, obs_max)

    obs_min = np.concatenate([env.workspace_min, [0., 0.],  *[env.workspace_min for _ in range(env.n)]], 0)
    obs_max = np.concatenate([env.workspace_max, [0.05, 0.05],  *[env.workspace_max for _ in range(env.n)]], 0)
    env = NormObsWrapper(env, obs_min, obs_max)
    # env = wrap_mega_env(env, info_to_obs)

    env = GymToGymnasiumSingle(env)
    
    return env

def episode_render_fn(env, ep):
    sim = env.sim
    all_img = []
    # reset the robot.
    env.reset()
    inner_env = env.env._env._env._env
    # move the robot arm out of the way
    if env.n == 2:
        out_of_way_state = np.array([ 4.40000000e+00,  4.04998318e-01,  4.79998255e-01,  3.11127168e-06,
            1.92819215e-02, -1.26133677e+00,  9.24837728e-02, -1.74551950e+00,
        -6.79993234e-01, -1.62616316e+00,  4.89490853e-01,  1.25022086e+00,
            2.02171933e+00, -2.35683450e+00,  8.60046276e-03, -6.44277362e-08,
            1.29999928e+00,  5.99999425e-01,  4.24784489e-01,  1.00000000e+00,
        -2.13882881e-07,  2.67353601e-07, -1.03622169e-15,  1.29999961e+00,
            8.99999228e-01,  4.24784489e-01,  1.00000000e+00, -2.95494240e-07,
            1.47747120e-07, -2.41072272e-15, -5.44202926e-07, -5.43454906e-07,
            7.61923038e-07,  5.39374476e-03,  1.92362793e-12,  7.54386574e-05,
            2.07866306e-04,  7.29063886e-03, -6.50353144e-03,  2.87876616e-03,
            8.29802372e-03, -3.06640616e-03, -1.17278073e-03,  2.71063610e-03,
        -1.62474545e-06, -1.60648093e-07, -1.28518475e-07,  1.09679929e-14,
            5.16300606e-06, -6.45375757e-06,  4.68203006e-17, -8.87786549e-08,
        -1.77557310e-07,  1.09035019e-14,  7.13305591e-06, -3.56652796e-06,
            6.54969586e-17])
    elif env.n == 3:
        out_of_way_state = np.array([4.40000000e+00,  4.04999349e-01,  4.79999636e-01,  2.79652104e-06,
        1.56722299e-02,-3.41500342e+00, 9.11469058e-02,-1.27681180e+00,
    -1.39750475e+00, 4.43858450e+00, 7.47892234e-01, 2.53633962e-01,
        2.34366216e+00, 3.35102418e+00, 8.32919575e-04, 1.41610111e-03,
        1.32999932e+00, 6.49999392e-01, 4.24784489e-01, 1.00000000e+00,
    -2.28652597e-07, 2.56090909e-07,-1.20181003e-15, 1.32999955e+00,
        8.49999274e-01, 4.24784489e-01, 1.00000000e+00,-2.77140579e-07,
        1.72443027e-07,-1.77971404e-15, 1.39999939e+00, 7.49999392e-01,
        4.24784489e-01, 1.00000000e+00,-2.31485576e-07, 2.31485577e-07,
    -6.68816586e-16,-4.48284993e-08,-8.37398903e-09, 7.56100615e-07,
        5.33433335e-03, 2.91848485e-01, 7.45623586e-05, 2.99902784e-01,
    -7.15601860e-02,-9.44665089e-02, 1.49646097e-02,-1.10990294e-01,
    -3.30174644e-03, 1.19462201e-01, 4.05130821e-04,-3.95036450e-04,
    -1.53880539e-07,-1.37393338e-07, 1.07636483e-14, 5.51953825e-06,
    -6.18188284e-06, 1.31307184e-17,-1.03617993e-07,-1.66528917e-07,
        1.06089030e-14, 6.69000941e-06,-4.16267252e-06, 3.63225324e-17,
    -1.39095626e-07,-1.39095626e-07, 1.10587840e-14, 5.58792469e-06,
    -5.58792469e-06,-2.07082526e-17])
    
    sim.set_state_from_flattened(out_of_way_state)
    sim.forward()
    inner_env.goal = ep['goal'][0]
    sites_offset = (sim.data.site_xpos - sim.model.site_pos)
    site_id = sim.model.site_name2id('gripper_site')
    def unnorm_ob(ob):
        return env.obs_min + ob * (env.obs_max -  env.obs_min)
    
    obs = ep['observation']
    obs = unnorm_ob(obs)
    grip_pos = obs[:3]
    gripper_state = obs[3:5]
    all_obj_pos = np.split(obs[5:5+3*env.n], env.n)
    # set the end effector site instead of the actual end effector.
    sim.model.site_pos[site_id] = grip_pos - sites_offset[site_id]
    # set the objects
    for i, pos in enumerate(all_obj_pos):
        sim.data.set_joint_qpos(f"object{i}:joint", [*pos, *[1,0,0,0]])

    sim.forward()
    img = sim.render(height=200, width=200, camera_name="external_camera_0")[::-1]
    
    return img

if __name__ == '__main__':
    timesteps = 100
    eval = False
    block_num = 3

    # env = get_env(timesteps, eval, block_num)
    env = get_pickplace_env(timesteps, eval)
    pdb.set_trace()
    reset_data = env.reset()
    step_data = env.step(np.array([0,0,0,0]))
    im_arr = env.render()
    plt.figure()
    plt.imshow(im_arr)
    plt.savefig('test.png')
    obs = step_data[0]
    episode_render_fn(env, obs)
    pdb.set_trace()
    print('o?')