import gym
import numpy as np
import pdb

# goal wrapper
class ConvertGoalEnvWrapper:
  """
  Given a GoalEnv that returns obs dict {'observation', 'achieved_goal', 'desired_goal'}, we modify obs dict to just contain {'observation', 'goal'} where 'goal' is desired goal.
  """
  def __init__(self, env, obs_key='observation', goal_key='goal'):
    self._env = env
    self.obs_key = obs_key
    self.goal_key = goal_key
    self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
    assert self._obs_is_dict, "GoalEnv should have obs dict"
    self._act_is_dict = hasattr(self._env.action_space, 'spaces')

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def step(self, action):
    # toss out achieved_goal
    obs, reward, done, info = self._env.step(action)
    obs = {self.obs_key: obs[self.obs_key], self.goal_key: obs['desired_goal']}
    return obs, reward, done, info

  def reset(self, **kwargs):
    # toss out achieved_goal and desired_goal keys.
    obs = self._env.reset()
    obs = {self.obs_key: obs[self.obs_key], self.goal_key: obs['desired_goal']}
    return obs

  @property
  def observation_space(self):
    # just return dict with observation.
    return gym.spaces.Dict({self.obs_key: self._env.observation_space[self.obs_key], self.goal_key: self._env.observation_space["desired_goal"]})

# normalize observation
class NormObsWrapper:
  # 1. assumes we have observation, achieved_goal, desired_goal with same dims.
  # 2. we don't guarantee that normed obs is between [0, 1], since obs_min / obs_max are arbitrary bounds.
  def __init__(self, env, obs_min, obs_max, keys=None):
    self._env = env
    self.obs_min = obs_min
    self.obs_max = obs_max
    self.keys = keys

  def __getattr__(self, name):
    return getattr(self._env, name)

  def norm_ob_dict(self, ob_dict):
    ob_dict = ob_dict.copy()
    if self.keys is None:
      for k, v in ob_dict.items():
        ob_dict[k] = (v -  self.obs_min) / (self.obs_max - self.obs_min)
    else:
      for k in self.keys:
        v = ob_dict[k]
        ob_dict[k] = (v -  self.obs_min) / (self.obs_max - self.obs_min)
    return ob_dict

  def step(self, action):
    obs, rew, done, info = self._env.step(action)
    return self.norm_ob_dict(obs), rew, done, info

  def reset(self, **kwargs):
    return self.norm_ob_dict(self._env.reset())

  def norm_ob(self, ob):
    return (ob - self.obs_min) / (self.obs_max - self.obs_min)

  def get_goals(self):
    goals = self._env.get_goals()
    norm_goals = np.stack([self.norm_ob(g) for g in goals])
    return norm_goals

class ClipObsWrapper:
    def __init__(self, env, obs_min, obs_max):
        self._env = env
        self.obs_min = obs_min
        self.obs_max = obs_max

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        new_obs = np.clip(obs['observation'], self.obs_min, self.obs_max)
        obs['observation'] = new_obs
        return obs, rew, done, info
    
class GymWrapper:
  """modifies obs space, action space,
  modifies step and reset fn, just returns ob dict now. ob dict has reward, done, etc.
  """
  def __init__(self, env, obs_key='image', act_key='action', info_to_obs_fn=None):
    self._env = env
    self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_is_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self.info_to_obs_fn = info_to_obs_fn

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    if self._obs_is_dict:
      spaces = self._env.observation_space.spaces.copy()
    else:
      spaces = {self._obs_key: self._env.observation_space}
    return {
        **spaces,
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }

  @property
  def act_space(self):
    if self._act_is_dict:
      return self._env.action_space.spaces.copy()
    else:
      return {self._act_key: self._env.action_space}

  def step(self, action):
    if not self._act_is_dict:
      action = action[self._act_key]
    obs, reward, done, info = self._env.step(action)
    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    obs['reward'] = float(reward)
    obs['is_first'] = False
    obs['is_last'] = done
    obs['is_terminal'] = info.get('is_terminal', done)
    if self.info_to_obs_fn:
      obs = self.info_to_obs_fn(info, obs)
    return obs

  def reset(self):
    obs = self._env.reset()
    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    obs['reward'] = 0.0
    obs['is_first'] = True
    obs['is_last'] = False
    obs['is_terminal'] = False
    if self.info_to_obs_fn:
      obs = self.info_to_obs_fn(None, obs)
    return obs
  
class OneHotAction:

  def __init__(self, env, key='action'):
    assert hasattr(env.act_space[key], 'n')
    self._env = env
    self._key = key
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    shape = (self._env.act_space[self._key].n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    space.n = shape[0]
    return {**self._env.act_space, self._key: space}

  def step(self, action):
    index = np.argmax(action[self._key]).astype(int)
    reference = np.zeros_like(action[self._key])
    reference[index] = 1
    if not np.allclose(reference, action[self._key]):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step({**action, self._key: index})

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.act_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference
  
class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.act_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return {**self._env.act_space, self._key: space}

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})