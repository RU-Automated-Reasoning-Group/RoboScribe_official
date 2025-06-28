import numpy as np
from gym.envs.robotics.fetch_env import FetchEnv, goal_distance
from gym.envs.robotics.fetch.push import FetchPushEnv
from gym.envs.robotics.fetch.slide import FetchSlideEnv
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.envs.robotics import fetch_env
import os
from gym.utils import EzPickle
from enum import Enum
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import block_diag

from gym.envs.robotics import rotations, utils
from gym import spaces

import pdb

dir_path = os.path.dirname(os.path.realpath(__file__))
STACKXML = os.path.join(dir_path, 'xmls', 'FetchStack#.xml')
ORIGPUSHXML = os.path.join(dir_path, 'xmls', 'Push.xml')
ORIGSLIDEXML = os.path.join(dir_path, 'xmls', 'Slide.xml')
PUSHXML = os.path.join(dir_path, 'xmls', 'CustomPush.xml')
PPXML = os.path.join(dir_path, 'xmls', 'CustomPP.xml')
SLIDEXML = os.path.join(dir_path, 'xmls', 'CustomSlide.xml')
SLIDE_N_XML = os.path.join(dir_path, 'xmls', 'FetchSlide#.xml')
PUSH_N_XML = os.path.join(dir_path, 'xmls', 'FetchPush#.xml')

INIT_Q_POSES = [
    [1.3, 0.6, 0.41, 1., 0., 0., 0.],
    [1.3, 0.9, 0.41, 1., 0., 0., 0.],
    [1.2, 0.68, 0.41, 1., 0., 0., 0.],
    [1.4, 0.82, 0.41, 1., 0., 0., 0.],
    [1.4, 0.68, 0.41, 1., 0., 0., 0.],
    [1.2, 0.82, 0.41, 1., 0., 0., 0.],
]
INIT_Q_POSES_SLIDE = [
    [1.3, 0.7, 0.42, 1., 0., 0., 0.],
    [1.3, 0.9, 0.42, 1., 0., 0., 0.],
    [1.25, 0.8, 0.42, 1., 0., 0., 0.],
    [1.35, 0.8, 0.42, 1., 0., 0., 0.],
    [1.35, 0.7, 0.42, 1., 0., 0., 0.],
    [1.25, 0.9, 0.42, 1., 0., 0., 0.],
]


class GoalType(Enum):
  OBJ = 1
  GRIP = 2
  OBJ_GRIP = 3
  ALL = 4
  OBJSPEED = 5
  OBJSPEED2 = 6
  OBJ_GRIP_GRIPPER = 7


def compute_reward(achieved_goal, goal, internal_goal, distance_threshold, per_dim_threshold,
                   compute_reward_with_internal, mode):
  # Always require internal success.
  internal_success = 0.
  if internal_goal == GoalType.OBJ_GRIP:
    if len(achieved_goal.shape) == 1:
      d = goal_distance(achieved_goal[:6], goal[:6])
    else:
      d = goal_distance(achieved_goal[:, :6], goal[:, :6])
  elif internal_goal in [GoalType.GRIP, GoalType.OBJ]:
    if len(achieved_goal.shape) == 1:
      d = goal_distance(achieved_goal[:3], goal[:3])
    else:
      d = goal_distance(achieved_goal[:, :3], goal[:, :3])
  elif internal_goal == GoalType.ALL:
    d = goal_distance(achieved_goal, goal)
  else:
    raise

  internal_success = (d <= distance_threshold).astype(np.float32)

  if compute_reward_with_internal:
    return internal_success - (1. - mode)

  # use per_dim_thresholds for other dimensions
  success = np.all(np.abs(achieved_goal - goal) < per_dim_threshold, axis=-1)
  success = np.logical_and(internal_success, success).astype(np.float32)
  return success - (1. - mode)


def get_obs(sim, external_goal, goal, subtract_obj_velp=True):
  # positions
  grip_pos = sim.data.get_site_xpos('robot0:grip')
  dt = sim.nsubsteps * sim.model.opt.timestep
  grip_velp = sim.data.get_site_xvelp('robot0:grip') * dt
  robot_qpos, robot_qvel = utils.robot_get_obs(sim)

  object_pos = sim.data.get_site_xpos('object0').ravel()
  # rotations
  object_rot = rotations.mat2euler(sim.data.get_site_xmat('object0')).ravel()
  # velocities
  object_velp = (sim.data.get_site_xvelp('object0') * dt).ravel()
  object_velr = (sim.data.get_site_xvelr('object0') * dt).ravel()
  # gripper state
  object_rel_pos = object_pos - grip_pos
  if subtract_obj_velp:
    object_velp -= grip_velp

  gripper_state = robot_qpos[-2:]
  gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

  items = [
      object_pos,
      grip_pos,
      object_rel_pos,
      gripper_state,
      object_rot,
      object_velp,
      object_velr,
      grip_velp,
      gripper_vel,
  ]

  obs = np.concatenate(items)

  if external_goal == GoalType.ALL:
    achieved_goal = np.concatenate([
        object_pos,
        grip_pos,
        object_rel_pos,
        gripper_state,
        object_rot,
        object_velp,
        object_velr,
        grip_velp,
        gripper_vel,
    ])
  elif external_goal == GoalType.OBJ:
    achieved_goal = object_pos
  elif external_goal == GoalType.OBJ_GRIP:
    achieved_goal = np.concatenate([object_pos, grip_pos])
  elif external_goal == GoalType.OBJ_GRIP_GRIPPER:
    achieved_goal = np.concatenate([object_pos, grip_pos, gripper_state])
  elif external_goal == GoalType.OBJSPEED:
    achieved_goal = np.concatenate([object_pos, object_velp])
  elif external_goal == GoalType.OBJSPEED2:
    achieved_goal = np.concatenate([object_pos, object_velp, object_velr])
  else:
    raise ValueError('unsupported goal type!')

  return {
      'observation': obs,
      'achieved_goal': achieved_goal,
      'desired_goal': goal.copy(),
  }


def sample_goal(initial_gripper_xpos, np_random, target_range, target_offset, height_offset, internal_goal,
                external_goal, grip_offset, gripper_goal):
  obj_goal = initial_gripper_xpos[:3] + np_random.uniform(-target_range, target_range, size=3)
  obj_goal += target_offset
  obj_goal[2] = height_offset

  if internal_goal in [GoalType.GRIP, GoalType.OBJ_GRIP]:
    grip_goal = initial_gripper_xpos[:3] + np_random.uniform(-0.15, 0.15, size=3) + np.array([0., 0., 0.15])
    obj_rel_goal = obj_goal - grip_goal
  else:
    grip_goal = obj_goal + grip_offset
    obj_rel_goal = -grip_offset

  if external_goal == GoalType.ALL:
    return np.concatenate([obj_goal, grip_goal, obj_rel_goal, gripper_goal, [0.] * 14])
  elif external_goal == GoalType.OBJ:
    return obj_goal
  elif external_goal == GoalType.OBJ_GRIP_GRIPPER:
    return np.concatenate([obj_goal, grip_goal, gripper_goal])
  elif external_goal == GoalType.OBJ_GRIP:
    return np.concatenate([obj_goal, grip_goal])
  elif external_goal == GoalType.OBJSPEED:
    return np.concatenate([obj_goal, [0.] * 3])
  elif external_goal == GoalType.OBJSPEED2:
    return np.concatenate([obj_goal, [0.] * 6])
  elif external_goal == GoalType.GRIP:
    raise NotImplementedError

  raise ValueError("BAD external goal value")


class StackEnv(fetch_env.FetchEnv, EzPickle):
  def __init__(self,
               max_step=50,
               n=1,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               hard=False,
               distance_threshold=0.,
               range_min=None,
               range_max=None):
    self.internal_goal = internal_goal
    self.external_goal = external_goal
    self.n = n
    self.hard = hard

    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
    }
    for i in range(self.n):
      k = 'object{}:joint'.format(i)
      initial_qpos[k] = INIT_Q_POSES[i]

    fetch_env.FetchEnv.__init__(self,
                                STACKXML.replace('#', '{}'.format(n)),
                                has_object=True,
                                block_gripper=False,
                                n_substeps=20,
                                gripper_extra_height=0.2,
                                target_in_the_air=False,
                                target_offset=0.0,
                                obj_range=0.15,
                                target_range=0.1,
                                distance_threshold=0.05,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')

    EzPickle.__init__(self)

    self.max_step = max(50 * (n - 1), 50)
    self.num_step = 0

    self.mode = 0
    if mode == "0/1" or mode == 1:
      self.mode = 1

    if self.external_goal == internal_goal:
      self.compute_reward_with_internal = True
    else:
      self.compute_reward_with_internal = compute_reward_with_internal

    self.per_dim_threshold = np.sqrt(self.distance_threshold**2 / 3)
    if per_dim_threshold:
      raise NotImplementedError

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.

    if self.external_goal == GoalType.OBJ_GRIP:
      if len(achieved_goal.shape) == 1:
        actual_internal_goals = np.split(goal[:-3], self.n)
        achieved_internal_goals = np.split(achieved_goal[:-3], self.n)
      else:
        actual_internal_goals = np.split(goal[:, :-3], self.n, axis=1)
        achieved_internal_goals = np.split(achieved_goal[:, :-3], self.n, axis=1)
    elif self.external_goal == GoalType.OBJ:
      if len(achieved_goal.shape) == 1:
        actual_internal_goals = np.split(goal, self.n)
        achieved_internal_goals = np.split(achieved_goal, self.n)
      else:
        actual_internal_goals = np.split(goal, self.n, axis=1)
        achieved_internal_goals = np.split(achieved_goal, self.n, axis=1)
    else:
      raise

    if len(achieved_goal.shape) == 1:
      success = 1.
    else:
      success = np.ones(achieved_goal.shape[0], dtype=np.float32)

    for b, g in zip(achieved_internal_goals, actual_internal_goals):
      d = goal_distance(b, g)
      success *= (d <= self.distance_threshold).astype(np.float32)

    if self.compute_reward_with_internal:
      return success - (1. - self.mode)

    # use per_dim_thresholds for other dimensions
    if len(achieved_goal.shape) == 1:
      d = goal_distance(achieved_goal[-3:], goal[-3:])
    else:
      d = goal_distance(achieved_goal[:, -3:], goal[:, -3:])
    success *= (d <= self.distance_threshold).astype(np.float32)

    return success - (1. - self.mode)

  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

    obj_feats = []
    obj_poses = []

    for i in range(self.n):
      obj_labl = 'object{}'.format(i)
      object_pos = self.sim.data.get_site_xpos(obj_labl).ravel()
      object_pos[2] = max(object_pos[2], self.height_offset)
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(obj_labl)).ravel()
      # velocities
      object_velp = (self.sim.data.get_site_xvelp(obj_labl) * dt).ravel()
      object_velr = (self.sim.data.get_site_xvelr(obj_labl) * dt).ravel()
      # gripper state
      object_rel_pos = object_pos - grip_pos
      #object_velp -= grip_velp

      obj_feats.append([object_pos, object_rel_pos, object_rot, object_velp, object_velr])
      obj_poses.append(object_pos)

    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
    if self.external_goal == GoalType.OBJ_GRIP:
      achieved_goal = np.concatenate(obj_poses + [grip_pos])
    else:
      achieved_goal = np.concatenate(obj_poses)
    obs = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel] + sum(obj_feats, []))

    return {
        'observation': obs,
        'achieved_goal': achieved_goal,
        'desired_goal': self.goal.copy(),
    }

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos)
    if self.external_goal == GoalType.OBJ_GRIP:
      goals = np.split(self.goal[:-3], self.n)
    else:
      goals = np.split(self.goal, self.n)

    for i in range(self.n):
      site_id = self.sim.model.site_name2id('target{}'.format(i))
      self.sim.model.site_pos[site_id] = goals[i] - sites_offset[i]
    self.sim.forward()

  def _reset_sim(self):
    self.sim.set_state(self.initial_state)

    # Only a little randomize about the start state
    for i in range(self.n):
      object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
      object_qpos[:2] += self.np_random.uniform(-0.03, 0.03, size=2)
      self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)

    bad_poses = [self.initial_gripper_xpos[:2]]
    # Randomize start positions of boxes.
    # for i in range(self.n):
    #   object_xpos = self.initial_gripper_xpos[:2]
    #   while min([np.linalg.norm(object_xpos - p) for p in bad_poses]) < 0.1:
    #       object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
    #   bad_poses.append(object_xpos)

    #   object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
    #   assert object_qpos.shape == (7,)
    #   object_qpos[:2] = object_xpos
    #   self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)

    self.sim.forward()
    return True

  def _sample_goal(self):
    bottom_box = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
    bottom_box[2] = self.height_offset  #self.sim.data.get_joint_qpos('object0:joint')[:3]

    goal = []
    for i in range(self.n):
      goal.append(bottom_box + (np.array([0., 0., 0.05]) * i))

    if self.external_goal == GoalType.OBJ_GRIP:
      goal.append(goal[-1] + np.array([-0.01, 0., 0.008]))

    return np.concatenate(goal)

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    if self.mode == 1 and reward:
      done = True

    info['is_success'] = np.allclose(reward, self.mode)
    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs


class PushEnv(FetchPushEnv):
  def __init__(self,
               max_step=51,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               hard=False,
               distance_threshold=0.,
               n=0,
               range_min=None,
               range_max=None):
    self.internal_goal = internal_goal
    self.external_goal = external_goal
    if hard or n > 0:
      raise ValueError("Hard not supported")
    super().__init__(reward_type='sparse')

    if distance_threshold > 1e-5:
      self.distance_threshold = distance_threshold

    if self.internal_goal == GoalType.OBJ_GRIP:
      self.distance_threshold *= np.sqrt(2)

    self.max_step = max_step
    self.num_step = 0
    self.mode = 0
    if mode == "0/1" or mode == 1:
      self.mode = 1

    if self.external_goal == internal_goal:
      self.compute_reward_with_internal = True
    else:
      self.compute_reward_with_internal = compute_reward_with_internal

    self.per_dim_threshold = np.sqrt(self.distance_threshold**2 / 3)
    if per_dim_threshold:
      self.per_dim_threshold = per_dim_threshold
    print('PER DIM THRESHOLD:', self.per_dim_threshold)

  def compute_reward(self, achieved_goal, goal, info):
    return compute_reward(achieved_goal, goal, self.internal_goal, self.distance_threshold, self.per_dim_threshold,
                          self.compute_reward_with_internal, self.mode)

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    goal = self.goal[:3]

    self.sim.model.site_pos[site_id] = goal - sites_offset[0]
    self.sim.forward()

  def _get_obs(self):
    return get_obs(self.sim, self.external_goal, self.goal)

  def _sample_goal(self):
    return sample_goal(self.initial_gripper_xpos, self.np_random, self.target_range, self.target_offset,
                       self.height_offset, self.internal_goal, self.external_goal, np.array([0., 0., 0.05]), [0., 0.])

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    if self.mode == 1 and reward:
      done = True

    info['is_success'] = np.allclose(reward, self.mode)
    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs


class SlideEnv(FetchSlideEnv):
  def __init__(self,
               max_step=51,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               hard=False,
               distance_threshold=0.,
               n=0,
               range_min=None,
               range_max=None):
    self.internal_goal = internal_goal
    self.external_goal = external_goal

    self.subtract_obj_velp = True
    if self.external_goal in [GoalType.OBJSPEED, GoalType.OBJSPEED2]:
      self.subtract_obj_velp = False

    if hard or n > 0:
      raise ValueError("Hard not supported")
    super().__init__(reward_type='sparse')

    if distance_threshold > 1e-5:
      self.distance_threshold = distance_threshold

    self.max_step = max_step
    self.num_step = 0
    self.mode = 0
    if mode == "0/1" or mode == 1:
      self.mode = 1

    if self.external_goal == internal_goal:
      self.compute_reward_with_internal = True
    else:
      self.compute_reward_with_internal = compute_reward_with_internal

    self.per_dim_threshold = np.sqrt(self.distance_threshold**2 / 3)
    if isinstance(per_dim_threshold, float) and per_dim_threshold > 1e-3:
      self.per_dim_threshold = per_dim_threshold
    print('PER DIM THRESHOLD:', self.per_dim_threshold)

  def compute_reward(self, achieved_goal, goal, info):
    return compute_reward(achieved_goal, goal, self.internal_goal, self.distance_threshold, self.per_dim_threshold,
                          self.compute_reward_with_internal, self.mode)

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    goal = self.goal[:3]

    self.sim.model.site_pos[site_id] = goal - sites_offset[0]
    self.sim.forward()

  def _get_obs(self):
    return get_obs(self.sim, self.external_goal, self.goal, self.subtract_obj_velp)

  def _sample_goal(self):
    return sample_goal(self.initial_gripper_xpos, self.np_random, self.target_range, self.target_offset,
                       self.height_offset, self.internal_goal, self.external_goal, np.array([0., 0., 0.05]), [0., 0.])

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    if self.mode == 1 and reward:
      done = True

    info['is_success'] = np.allclose(reward, self.mode)
    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs


class PickPlaceEnv(FetchPickAndPlaceEnv):
  def __init__(self,
               max_step=51,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               hard=False,
               distance_threshold=0.,
               n=0.5,
               range_min=0.2,
               range_max=0.45,
               gripper_extra_height=0.2,
               obj_range=0.15,
               target_range=0.15):
    self.internal_goal = internal_goal
    self.external_goal = external_goal
    if hard:
      self.minimum_air = range_min
      self.maximum_air = range_max
    else:
      self.minimum_air = 0.
      self.maximum_air = range_max
    self.in_air_percentage = n
    # super().__init__(reward_type='sparse')
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
    }
    fetch_env.FetchEnv.__init__(
        self, PPXML, has_object=True, block_gripper=False, n_substeps=20,
        gripper_extra_height=gripper_extra_height, target_in_the_air=True, target_offset=0.0,
        obj_range=obj_range, target_range=target_range, distance_threshold=0.05,
        initial_qpos=initial_qpos, reward_type="sparse")
    EzPickle.__init__(self)

    if distance_threshold > 1e-5:
      self.distance_threshold = distance_threshold

    self.max_step = max_step
    self.num_step = 0
    self.mode = 0
    if mode == "0/1" or mode == 1:
      self.mode = 1

    if self.external_goal == internal_goal:
      self.compute_reward_with_internal = True
    else:
      self.compute_reward_with_internal = compute_reward_with_internal

    self.per_dim_threshold = np.sqrt(self.distance_threshold**2 / 3)
    if per_dim_threshold:
      self.per_dim_threshold = per_dim_threshold
    print('PER DIM THRESHOLD:', self.per_dim_threshold)

  def compute_reward(self, achieved_goal, goal, info):
    return compute_reward(achieved_goal, goal, self.internal_goal, self.distance_threshold, self.per_dim_threshold,
                          self.compute_reward_with_internal, self.mode)

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')
    obj_goal = self.goal[:3]
    self.sim.model.site_pos[site_id] = obj_goal - sites_offset[0]
    if self.internal_goal == GoalType.ALL:
      site_id = self.sim.model.site_name2id('target1')
      grip_goal = self.goal[3:6]
      self.sim.model.site_pos[site_id] = grip_goal - sites_offset[0]
    self.sim.forward()

  def _get_obs(self):
    return get_obs(self.sim, self.external_goal, self.goal)

  def _sample_goal(self):
    height_offset = self.height_offset
    if self.np_random.uniform() < self.in_air_percentage:
      height_offset += self.np_random.uniform(self.minimum_air, self.maximum_air)
    return sample_goal(self.initial_gripper_xpos, self.np_random, self.target_range, self.target_offset, height_offset,
                       self.internal_goal, self.external_goal, np.array([-0.01, 0., 0.008]), [0.024273, 0.024273])

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    if self.mode == 1 and reward:
      done = True

    info['is_success'] = np.allclose(reward, self.mode)
    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

  def get_metrics_dict(self):
    info = {"is_success": float(False)}
    return info

class DemoStackEnv(fetch_env.FetchEnv, EzPickle):
  def __init__(self,
               max_step=50,
               n=2,
               mode="-1/0",
               hard=False,
               distance_threshold=0.03,
               eval=False,
               xml=STACKXML,
               workspace_min=None,
               workspace_max=None,
               initial_qpos=None):
    self.n = n
    self.hard = hard
    self.distance_threshold = distance_threshold
    self.eval = eval
    if workspace_min is None:
      workspace_min=np.array([1.25, 0.5, 0.42])
    if workspace_max is None:
      workspace_max=np.array([1.6, 1.0, 0.6])

    self.workspace_min = workspace_min
    self.workspace_max = workspace_max
    if initial_qpos is None:
      initial_qpos = {
          'robot0:slide0': 0.405,
          'robot0:slide1': 0.48,
          'robot0:slide2': 0.0,
          'object0:joint': [1.3, 0.6, 0.41, 1., 0., 0., 0.],
          'object1:joint': [1.3, 0.9, 0.41, 1., 0., 0., 0.],
      }
    self.initial_qpos = initial_qpos

    self.all_goals = self._create_goals()
    self.goal_idx = -1

    fetch_env.FetchEnv.__init__(self,
                                xml.replace('#', '{}'.format(n)),
                                has_object=True,
                                block_gripper=False,
                                n_substeps=20,
                                gripper_extra_height=0.2,
                                target_in_the_air=False,
                                target_offset=0.0,
                                obj_range=0.15,
                                target_range=0.0,
                                distance_threshold=distance_threshold,
                                initial_qpos=self.initial_qpos,
                                reward_type='sparse')

    EzPickle.__init__(self)

    self.max_step = max_step
    self.num_step = 0

    self.mode = 0
    if mode == "0/1" or mode == 1:
      self.mode = 1

  def _create_goals(self):
    gripper_offset = np.array([-0.01, 0, 0.008])

    """     g
            2
            1
    """
    final_goal_1 = np.array([1.33193233, 0.74910037, 0.48273329, 0.05 ,  0.05, 1.34193233, 0.74910037, 0.42473329, 1.34193233, 0.74910037, 0.47473329])
    """     g
            1
            2
    """
    temp = np.copy(final_goal_1)
    final_goal_2 = np.copy(final_goal_1)
    final_goal_2[8:11] = final_goal_2[5:8]
    final_goal_2[5:8] = temp[8:11]

    obj0_init_pos = self.initial_qpos['object0:joint'][:3]
    obj1_init_pos = self.initial_qpos['object1:joint'][:3]

    """ g
        0       1
    gripper over first block.
    """
    grip_pos = np.copy(obj0_init_pos) + gripper_offset
    gripper_state = [0.03, 0.03]
    goal_1 = np.concatenate([grip_pos, gripper_state, obj0_init_pos, obj1_init_pos])

    """         g
        0       1
    gripper over 2nd block.
    """
    grip_pos = np.copy(obj1_init_pos) + gripper_offset
    gripper_state = [0.03, 0.03]
    goal_2 = np.concatenate([grip_pos, gripper_state, obj0_init_pos, obj1_init_pos])

    """    g
           0
                 1
    gripper pick first block.
    """
    obj0_lifted_pos = obj0_init_pos + np.array([0, 0, 0.05])
    grip_pos = obj0_lifted_pos + gripper_offset
    gripper_state = [0.0, 0.0]
    goal_3 = np.concatenate([grip_pos, gripper_state, obj0_lifted_pos, obj1_init_pos])

    """    g
           1
       0
    gripper pick second block.
    """
    obj1_lifted_pos = obj1_init_pos + np.array([0, 0, 0.05])
    grip_pos = obj1_lifted_pos + gripper_offset
    gripper_state = [0.0, 0.0]
    goal_4 = np.concatenate([grip_pos, gripper_state, obj0_init_pos, obj1_lifted_pos])


    return np.stack([goal_1, goal_2, goal_3, goal_4, final_goal_1, final_goal_2])


  def compute_reward(self, achieved_goal, goal, info):
    ag_poses = np.split(achieved_goal[5:], self.n)
    # ag_poses.append(achieved_goal[:3])
    ag_poses = np.array(ag_poses)

    goal_poses = np.split(goal[5:], self.n)
    # goal_poses.append(goal[:3])
    goal_poses = np.array(goal_poses)

    dist_per_obj = np.linalg.norm(ag_poses - goal_poses, axis=1)
    succ_per_obj = dist_per_obj < self.distance_threshold
    all_succ = np.all(succ_per_obj).astype(np.float32)
    reward = all_succ - 1 # maps to -1 if fail, 0 if success.
    return reward
  
  def compute_reward_detail(self, achieved_goal, goal, info):
    ag_poses = np.split(achieved_goal[5:], self.n)
    # ag_poses.append(achieved_goal[:3])
    ag_poses = np.array(ag_poses)

    goal_poses = np.split(goal[5:], self.n)
    # goal_poses.append(goal[:3])
    goal_poses = np.array(goal_poses)

    dist_per_obj = np.linalg.norm(ag_poses - goal_poses, axis=1)
    # succ_per_obj = dist_per_obj < self.distance_threshold
    # all_succ = np.all(succ_per_obj).astype(np.float32)
    # reward = all_succ - 1 # maps to -1 if fail, 0 if success.
    return dist_per_obj

  def _get_obs(self):
    # just return grip pos, n obj pos.
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
    gripper_state = robot_qpos[-2:]
    obj_poses = []
    for i in range(self.n):
      obj_labl = 'object{}'.format(i)
      object_pos = self.sim.data.get_site_xpos(obj_labl).ravel()
      object_pos[2] = max(object_pos[2], self.height_offset)
      obj_poses.append(object_pos)
    obj_poses = np.concatenate(obj_poses)
    achieved_goal = np.concatenate([grip_pos, gripper_state, obj_poses])
    obs = achieved_goal.copy()
    return {
        'observation': obs,
        'achieved_goal': achieved_goal,
        'desired_goal': self.goal.copy(),
    }

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos)
    goals = np.split(self.goal[5:], self.n)
    for i in range(self.n):
      site_id = self.sim.model.site_name2id('target{}'.format(i))
      self.sim.model.site_pos[site_id] = goals[i] - sites_offset[site_id]
    grip_pos = self.goal[:3]
    site_id = self.sim.model.site_name2id('gripper_site')
    self.sim.model.site_pos[site_id] = grip_pos - sites_offset[site_id]
    self.sim.forward()

  def _reset_sim(self):
    self.sim.set_state(self.initial_state)

    # Only a little randomize about the start state
    # for i in range(self.n):
      # object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
      # object_qpos[:2] += self.np_random.uniform(-0.03, 0.03, size=2)
      # self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)

    # bad_poses = [self.initial_gripper_xpos[:2]]
    # Randomize start positions of boxes.
    # for i in range(self.n):
    #   object_xpos = self.initial_gripper_xpos[:2]
    #   while min([np.linalg.norm(object_xpos - p) for p in bad_poses]) < 0.1:
    #       object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
    #   bad_poses.append(object_xpos)

    #   object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
    #   assert object_qpos.shape == (7,)
    #   object_qpos[:2] = object_xpos
    #   self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)

    self.sim.forward()
    return True

  # def _sample_goal(self):
  #   bottom_box = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
  #   bottom_box[2] = self.height_offset  #self.sim.data.get_joint_qpos('object0:joint')[:3]
  #   gripper_state = np.array([-0.02, 0.02]) # Assume gripper state as open as possible
  #   obj_pos = []
  #   for i in range(self.n):
  #     obj_pos.append(bottom_box + (np.array([0., 0., 0.05]) * i))
  #   grip_pos = obj_pos[-1] + np.array([-0.01, 0., 0.008])
  #   obj_pos = np.concatenate(obj_pos)
  #   goal = np.concatenate([grip_pos, gripper_state, obj_pos])
  #   return goal

  def set_goal_idx(self, idx):
    self.goal_idx = idx

  def get_goal_idx(self):
    return self.goal_idx

  def get_goals(self):
    return self.all_goals

  def get_metrics_dict(self):
    info = {"is_success": float(False)}
    return info

  def _sample_goal(self):
    if self.goal_idx < 9 or self.goal_idx > 14:
      self.goal_idx = 9
    else:
      self.goal_idx += 1
      if self.goal_idx > 14:
        self.goal_idx = 9
    # print("goal key", self.goal_idx)
    return self.all_goals[self.goal_idx]

  def step(self, action):
    # check if action is out of bounds
    action = action.copy()
    curr_eef_state = self.sim.data.get_site_xpos('robot0:grip')
    next_eef_state = curr_eef_state + (action[:3] * 0.05)

    next_eef_state = np.clip(next_eef_state, self.workspace_min, self.workspace_max)
    clipped_ac = (next_eef_state - curr_eef_state) / 0.05
    action[:3] = clipped_ac

    obs, reward, _, info = super().step(action)
    self.num_step += 1

    if self.eval:
      done = np.allclose(0., reward)
      # info['is_success'] = done
      info = self.add_pertask_success(info, obs['observation'], goal_idx=self.goal_idx)
    else:
      done = False
      # info['is_success'] = np.allclose(0., reward)
      info = self.add_pertask_success(info, obs['observation'], goal_idx=None)

    all_obj_poses = np.split(obs['observation'][5:], self.n)
    z_threshold = 0.5
    for idx, obj_pos in enumerate(all_obj_poses):
      info[f"metric_obj{idx}_above_{z_threshold:.2f}"] = float(obj_pos[2] > z_threshold)

    done = True if self.num_step >= self.max_step else done
    if done: info['TimeLimit.truncated'] = True

    return obs, reward, done, info

  def add_pertask_success(self, info, obs, goal_idx = None):
    goal_idxs = [goal_idx] if goal_idx is not None else range(len(self.all_goals))
    for goal_idx in goal_idxs:
      g = self.all_goals[goal_idx]
      # compute normal success - if we reach within 0.15
      reward = self.compute_reward(obs, g, info)
      # -1 if not close, 0 if close.
      # map to 0 if not close, 1 if close.
      info[f"metric_success/goal_{goal_idx}"] = reward + 1
    return info

  def get_metrics_dict(self):
    info = {}
    dummy_obs = np.ones(self.observation_space['achieved_goal'].shape)
    if self.eval:
      info = self.add_pertask_success(info, dummy_obs, goal_idx=self.goal_idx)
      # by default set it to false.
      info[f"metric_success/goal_{self.goal_idx}"] = 0.0
    else:
      info = self.add_pertask_success(info, dummy_obs, goal_idx=None)
      for k,v in info.items():
        if 'metric' in k:
          info[k] = 0.0
    z_threshold = 0.5
    for idx in range(self.n):
      info[f"metric_obj{idx}_above_{z_threshold:.2f}"] = 0.0
    return info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

  def render(self, mode='human', width=100, height=100):
      self._render_callback()
      if mode == 'rgb_array':
          return self.sim.render(height=100, width=100, camera_name="external_camera_0")[::-1]
      
          # self._get_viewer(mode).render(width, height)
          # # window size used for old mujoco-py:
          # data = self._get_viewer(mode).read_pixels(width, height, depth=False)
          # # original image is upside-down, so flip it
          # return data[::-1, :, :]
      
          # pdb.set_trace()
          # self._get_viewer(mode).render(width, height)
          # # window size used for old mujoco-py:
          # data = self._get_viewer(mode).read_pixels(
          #     width, height, depth=False
          # )
          # original image is upside-down, so flip it
          # return data[::-1, :, :]

      elif mode == 'human':
          self._get_viewer(mode).render()

class WallsDemoSingleStackEnv(DemoStackEnv):
  def __init__(self,
               max_step=100,
               mode="-1/0",
               hard=False,
               distance_threshold=0.03,
               eval=False,
               initial_qpos=None):
    xml = os.path.join(dir_path, 'xmls', 'FetchStack#Walls.xml')
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.33, 0.65, 0.42, 1., 0., 0., 0.]
    }
    workspace_min=np.array([1.30, 0.64, 0.42])
    workspace_max=np.array([1.36, 0.86, 0.59])

    # candidate initial position for block
    self.candidate_init_pos = [[1.33, 0.65, 0.42, 1., 0., 0., 0.], 
                               [1.33, 0.85, 0.42, 1., 0., 0., 0.],
                               [1.33, 0.75, 0.42, 1., 0., 0., 0.]]

    super().__init__(
      max_step=max_step,
      n=1,
      mode=mode,
      hard=hard,
      distance_threshold=distance_threshold,
      eval=eval,
      xml=xml,
      workspace_min=workspace_min,
      workspace_max=workspace_max,
      initial_qpos=initial_qpos
    )

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

  def set_init_pos(self, idx):
    # self.initial_qpos['object0:joint'] = np.array(self.candidate_init_pos[idx])
    self.initial_state.qpos[-7:] = np.array(self.candidate_init_pos[idx])

  def _create_goals(self):

    cand_height = [0.426, 0.475, 0.525]
    cand_y = [0.65, 0.85, 0.75]
    all_goals = []

    for h in cand_height:
      for y in cand_y:
        block_goal = np.array([1.33, y, h])
        grip_goal = block_goal + np.array([-0.01, 0., 0.008])
        all_goals.append(np.expand_dims(np.concatenate([grip_goal, np.array([0.05, 0.05]), block_goal]), axis=0))

    return np.concatenate(all_goals, axis=0)


class WallsDemoStackEnv(DemoStackEnv):
  def __init__(self,
               max_step=100,
               n=2,
               mode="-1/0",
               hard=False,
               distance_threshold=0.03,
               eval=False,
               initial_qpos=None):
    xml = os.path.join(dir_path, 'xmls', 'FetchStack#Walls.xml')
    workspace_min=np.array([1.25, 0.55, 0.42])
    workspace_max=np.array([1.45, 0.95, 0.55])
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.33, 0.65, 0.42, 1., 0., 0., 0.],
        'object1:joint': [1.33, 0.85, 0.42, 1., 0., 0., 0.],
    }
    if n == 3:
      # initial_qpos['object2:joint']= [1.42, 0.75, 0.42, 1., 0., 0., 0.]
      # workspace_max=np.array([1.45, 0.95, 0.59])
      # 2D version
      initial_qpos['object2:joint']= [1.33, 0.75, 0.42, 1., 0., 0., 0.]
      workspace_min=np.array([1.30, 0.64, 0.42])
      workspace_max=np.array([1.36, 0.86, 0.59])

    super().__init__(
      max_step=max_step,
      n=n,
      mode=mode,
      hard=hard,
      distance_threshold=distance_threshold,
      eval=eval,
      xml=xml,
      workspace_min=workspace_min,
      workspace_max=workspace_max,
      initial_qpos=initial_qpos
    )
  def _create_goals(self):
    # gripper_offset = np.array([-0.01, 0, 0.008])
    # if self.n == 2:

    #   """     g
    #           2
    #           1
    #   """
    #   hard_stack_1 = np.array([1.33193233, 0.74910037, 0.48273329, 0.05 ,  0.05, 1.33193233, 0.74910037, 0.42473329, 1.33193233, 0.74910037, 0.47473329])
    #   """     g
    #           1
    #           2
    #   """
    #   temp = np.copy(hard_stack_1)
    #   hard_stack_2 = np.copy(hard_stack_1)
    #   hard_stack_2[8:11] = hard_stack_2[5:8]
    #   hard_stack_2[5:8] = temp[8:11]

    #   obj0_init_pos = self.initial_qpos['object0:joint'][:3]
    #   obj1_init_pos = self.initial_qpos['object1:joint'][:3]
    #   obj0_init_pos[2] = obj1_init_pos[2] = 0.425

    #   """ g
    #       0       1
    #   gripper over first block.
    #   """
    #   grip_pos = np.copy(obj0_init_pos) + gripper_offset
    #   gripper_state = [0.03, 0.03]
    #   touch_1 = np.concatenate([grip_pos, gripper_state, obj0_init_pos, obj1_init_pos])

    #   """         g
    #       0       1
    #   gripper over 2nd block.
    #   """
    #   grip_pos = np.copy(obj1_init_pos) + gripper_offset
    #   gripper_state = [0.03, 0.03]
    #   touch_2 = np.concatenate([grip_pos, gripper_state, obj0_init_pos, obj1_init_pos])
    #   """    g
    #         0
    #               1
    #   gripper pick first block.
    #   """
    #   obj0_lifted_pos = obj0_init_pos + np.array([0, 0.05, 0.1])
    #   grip_pos = obj0_lifted_pos + gripper_offset
    #   gripper_state = [0.0, 0.0]
    #   pick_1 = np.concatenate([grip_pos, gripper_state, obj0_lifted_pos, obj1_init_pos])

    #   """    g
    #         1
    #     0
    #   gripper pick second block.
    #   """
    #   obj1_lifted_pos = obj1_init_pos + np.array([0, -0.05, 0.1])
    #   grip_pos = obj1_lifted_pos + gripper_offset
    #   gripper_state = [0.0, 0.0]
    #   pick_2 = np.concatenate([grip_pos, gripper_state, obj0_init_pos, obj1_lifted_pos])

    #   """    g
    #         1
    #         0
    #   stack on first block
    #   """
    #   stack_1 = np.copy(hard_stack_1)
    #   stack_1[[1,6,9]] = obj0_init_pos[1]

    #   """    g
    #         0
    #         1
    #   stack on second block.
    #   """
    #   stack_2 = np.copy(hard_stack_2)
    #   stack_2[[1,6,9]] = obj1_init_pos[1]
    #   return np.stack([pick_1, pick_2, stack_1, stack_2, hard_stack_1, hard_stack_2])
    # elif self.n == 3:
    #   obj0_init_pos = self.initial_qpos['object0:joint'][:3]
    #   obj1_init_pos = self.initial_qpos['object1:joint'][:3]
    #   obj2_init_pos = self.initial_qpos['object2:joint'][:3]
    #   obj0_init_pos[2] = obj1_init_pos[2] = obj2_init_pos[2] = 0.425

    #   """ g
    #       0       1
    #   gripper over first block.
    #   """
    #   grip_pos = np.copy(obj0_init_pos) + gripper_offset
    #   gripper_state = [0.03, 0.03]
    #   touch_1 = np.concatenate([grip_pos, gripper_state, obj0_init_pos, obj1_init_pos, obj2_init_pos])

    #   """         g
    #       0       1
    #   gripper over 2nd block.
    #   """
    #   grip_pos = np.copy(obj1_init_pos) + gripper_offset
    #   gripper_state = [0.03, 0.03]
    #   touch_2 = np.concatenate([grip_pos, gripper_state, obj0_init_pos, obj1_init_pos, obj2_init_pos])
    #   """    g
    #         0
    #               1
    #   gripper pick first block.
    #   """
    #   obj0_lifted_pos = obj0_init_pos + np.array([0, 0.05, 0.1])
    #   grip_pos = obj0_lifted_pos + gripper_offset
    #   gripper_state = [0.0, 0.0]
    #   pick_1 = np.concatenate([grip_pos, gripper_state, obj0_lifted_pos, obj1_init_pos, obj2_init_pos])

    #   """    g
    #          1
    #     0
    #   gripper pick second block.
    #   """
    #   obj1_lifted_pos = obj1_init_pos + np.array([0, -0.05, 0.1])
    #   grip_pos = obj1_lifted_pos + gripper_offset
    #   gripper_state = [0.0, 0.0]
    #   pick_2 = np.concatenate([grip_pos, gripper_state, obj0_init_pos, obj1_lifted_pos, obj2_init_pos])

    #   """g
    #      1
    #      0
    #      2
    #   """
    #   obj2_pos = np.copy(obj2_init_pos)
    #   # obj2_pos[0] = obj0_init_pos[0]

    #   obj0_pos = np.copy(obj2_pos)
    #   obj0_pos[2] += 0.05
    #   obj1_pos = np.copy(obj0_pos)
    #   obj1_pos[2] += 0.05
    #   grip_pos = np.copy(obj1_pos) + gripper_offset
    #   gripper_state = [0.05, 0.05]
    #   final_stack_1 = np.concatenate([grip_pos, gripper_state, obj0_pos, obj1_pos, obj2_pos])

    #   harder_final_stack = np.copy(final_stack_1)
    #   harder_final_stack[:3] = [1.33, 0.75, 0.59]

    #   # return np.stack([touch_2,  pick_2, final_stack_1, harder_final_stack])
    #   """
    #   new extra goals. green always at bottom.
    #   """      
    #   obj2_pos = np.copy(obj0_init_pos)
    #   obj0_pos = np.copy(obj2_pos)
    #   obj0_pos[2] += 0.05
    #   obj1_pos = np.copy(obj0_pos)
    #   obj1_pos[2] += 0.05
    #   grip_pos = np.copy(obj1_pos) + gripper_offset
    #   gripper_state = [0.05, 0.05]
    #   final_stack_2 = np.concatenate([grip_pos, gripper_state, obj0_pos, obj1_pos, obj2_pos])

    #   obj2_pos = np.copy(obj1_init_pos)
    #   obj0_pos = np.copy(obj2_pos)
    #   obj0_pos[2] += 0.05
    #   obj1_pos = np.copy(obj0_pos)
    #   obj1_pos[2] += 0.05
    #   grip_pos = np.copy(obj1_pos) + gripper_offset
    #   gripper_state = [0.05, 0.05]
    #   final_stack_3 = np.concatenate([grip_pos, gripper_state, obj0_pos, obj1_pos, obj2_pos])

    #   return np.stack([final_stack_1, final_stack_2, final_stack_3])

    gripper_offset = np.array([-0.01, 0, 0.025])
    example_stack = np.array([1.33193233, 0.74910037, 0.52473329 + 0.0008, 0.05 ,  0.05, 1.33193233, 0.74910037, 0.42473329, 1.33193233, 0.74910037, 0.47473329, 1.33193233, 0.74910037, 0.52473329])
    all_goals = []
    # create reaching goals
    # for i in range(self.n):
    #   # reaching goal, reach i-th block
    #   goal = np.copy(example_stack)
    #   for j in range(3):
    #     start = 5 + (3 * j)
    #     goal[start: start+3] = self.initial_qpos[f"object{j}:joint"][:3]
    #   start = 5 + (3*i)
    #   goal[:3] = goal[start: start+3] + gripper_offset # place hand over i-th obj
    #   all_goals.append(goal)
    obj0_init_pos = self.initial_qpos['object0:joint'][:3]
    obj1_init_pos = self.initial_qpos['object1:joint'][:3]
    obj2_init_pos = self.initial_qpos['object2:joint'][:3]
    obj0_init_pos[2] = obj1_init_pos[2] = obj2_init_pos[2] = 0.425
    obj_init_pos = [obj0_init_pos, obj1_init_pos, obj2_init_pos]

    # create picking goals
    for i in range(self.n):
      # reaching goal, reach i-th block
      goal = np.copy(example_stack)
      for j in range(3):
        start = 5 + (3 * j)
        goal[start: start+3] = obj_init_pos[j][:3]
      start = 5 + (3*i)
      goal[start: start+3] = [1.34193271, 0.74910037, 0.53472273] # pick object to start.
      goal[:3] = goal[start: start+3] + np.array([-0.01, 0, 0.008]) # pick over i-th obj
      all_goals.append(goal)

    all_goals = np.stack(all_goals)

    top_stack_goals = []
    start_stack_goals = []
    remaining_stack_goals = []

    # from 1<=j<=N, generate N-j height stacks.
    from itertools import permutations
    for j in range(1,self.n):
      # need to try all permutations of blocks.
      for perm in permutations(range(self.n)):
        perm = list(perm)
        goal = np.copy(example_stack)
        # first j blocks set to initial pos.
        for i in range(j):
          start = 5 + (3 * perm[i])
          goal[start: start+3] = prev_pos = obj_init_pos[perm[i]][:3]
        # remaining blocks set on top.
        for i in range(j, self.n):
          start = 5 + (3 * perm[i])
          goal[start: start+3] = prev_pos = prev_pos + np.array([0,0,0.05])

        # put gripper on remaining block. (only for 2-height tower case.)
        if j == 2:
          intermediate_goal = np.copy(goal)
          start = 5 + (3 * perm[0])
          intermediate_goal[:3] = goal[start: start+3] + np.array([-0.01, 0, 0.008])
          remaining_stack_goals.append(intermediate_goal)

        # put gripper on top block.
        intermediate_goal = np.copy(goal)
        intermediate_goal[:3] = prev_pos + np.array([-0.01, 0, 0.008])
        top_stack_goals.append(intermediate_goal)

        # put gripper to start.
        end_goal = np.copy(goal)
        end_goal[:3] = [1.34193271, 0.74910037, 0.53472273] # move to start
        start_stack_goals.append(end_goal)

    # start_stack_goals = np.stack(start_stack_goals[::-1]) # 12
    # top_stack_goals = np.stack(top_stack_goals[::-1]) # 12
    # remaining_stack_goals = np.stack(remaining_stack_goals[::-1]) # 6
    # return np.concatenate([start_stack_goals])

    # all goals contains pick block to the middle goals. 3 of them.
    # top stack goals contain 2-3 stack towers with gripper on top block. First 6 are 3 height, next 6 are 2 height.
    # start stack goals are  2-3 stack towers with gripper at start pos. These are hard.
    # remaining_stack_goals are  2 stack twoers with gripper on the solitary block. 6 of them

    # 3 + 6 + 6 + 6 = 21 goals total.
    # level 1 will have 3.
    # level 2 will have 12.
    # level 3 will have 6.
    # return np.concatenate([all_goals, top_stack_goals[6:],  remaining_stack_goals, top_stack_goals[:6]])
    return np.concatenate([all_goals, top_stack_goals[6:], top_stack_goals[:6]])

    # return np.concatenate([all_goals, remaining_stack_goals, top_stack_goals, start_stack_goals])


class NoisyWallsDemoStackEnv(WallsDemoStackEnv):
  def __init__(self,
              noise_dim,
              noise_low,
              noise_high,
              reset_interval=1,
              max_step=100,
              n=2,
              mode="-1/0",
              hard=False,
              distance_threshold=0.03,
              eval=False,
              initial_qpos=None):
    super().__init__(max_step,n,mode,hard, distance_threshold, eval, initial_qpos)
    self.noise_dim = noise_dim
    self.noise_low = noise_low
    self.noise_high = noise_high
    obspace = self.observation_space
    self.observation_space = spaces.Dict(dict(
        desired_goal=spaces.Box(-np.inf, np.inf, shape=(obspace['desired_goal'].shape[0] + noise_dim,), dtype='float32'),
        achieved_goal=spaces.Box(-np.inf, np.inf, shape=(obspace['achieved_goal'].shape[0] + noise_dim,), dtype='float32'),
        observation=spaces.Box(-np.inf, np.inf, shape=(obspace['observation'].shape[0] + noise_dim,), dtype='float32'),
    ))
    self.reset_interval = reset_interval
    self.num_resets = 0

  def add_noise(self, obs):
    for k, v in obs.items():
      obs[k] = np.hstack([v, self.sampled_noise])

  def reset(self):
    print(self.num_resets)
    if self.num_resets % self.reset_interval == 0:
      print('generating noise')
      self.sampled_noise = np.random.uniform(low=self.noise_low, high=self.noise_high, size=self.noise_dim)
    self.num_resets += 1
    obs = super().reset()
    self.add_noise(obs)
    return obs

  def step(self, action):
    obs, rew, done, info = super().step(action)
    self.add_noise(obs)
    return obs, rew, done, info

  def get_goals(self):
    goals = super().get_goals()
    sampled_noise_mean = np.ones((goals.shape[0], self.noise_dim)) * (self.noise_low + self.noise_high)/2.0
    goals = np.hstack([goals, sampled_noise_mean])
    return goals


  def get_metrics_dict(self):
    info = {}
    dummy_obs = np.ones(self.observation_space['achieved_goal'].shape[0] - self.noise_dim)
    if self.eval:
      info = self.add_pertask_success(info, dummy_obs, goal_idx=self.goal_idx)
      # by default set it to false.
      info[f"metric_success/goal_{self.goal_idx}"] = 0.0
    else:
      info = self.add_pertask_success(info, dummy_obs, goal_idx=None)
      for k,v in info.items():
        if 'metric' in k:
          info[k] = 0.0
    z_threshold = 0.5
    for idx in range(self.n):
      info[f"metric_obj{idx}_above_{z_threshold:.2f}"] = 0.0
    return info


class DiscreteWallsDemoStackEnv(WallsDemoStackEnv):
  def __init__(self, max_step=100, n=2, mode="-1/0", hard=False, distance_threshold=0.03, eval=False, increment=0.01):
    """
    0/1: +- x
    2/3: +- y
    4/5: +- z
    6: toggle gripper
    """
    self._increment = increment
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.33, 0.6, 0.41, 1., 0., 0., 0.],
        'object1:joint': [1.33, 0.9, 0.41, 1., 0., 0., 0.],
    }
    super().__init__(max_step, n, mode, hard, distance_threshold, eval, initial_qpos=initial_qpos)
    self._close_gripper = True
    self.cont_action_space = self.action_space
    self.action_space = spaces.Discrete(7)

  def reset(self):
    self._close_gripper = False
    super().reset()
    # open the gripper.
    action = np.array([0, 0, 0, 1], dtype=np.float32)
    self._set_action(action)
    self.sim.step()
    self._step_callback()
    obs = self._get_obs()
    return obs

  def step(self, disc_action):
    pos_delta = [0, 0, 0]
    num_steps = 1
    if disc_action < 6:
      pos_delta = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]][disc_action]
    elif disc_action == 6:
      self._close_gripper = not self._close_gripper
      num_steps = 2

    gripper_value = -1 if self._close_gripper else 1
    action = np.array([*pos_delta, gripper_value], dtype=np.float32)
    ###### DemoStackEnv start ######
    action = action.copy()
    curr_eef_state = self.sim.data.get_site_xpos('robot0:grip')
    next_eef_state = curr_eef_state + (action[:3] * self._increment)

    next_eef_state = np.clip(next_eef_state, self.workspace_min, self.workspace_max)
    clipped_ac = (next_eef_state - curr_eef_state) / self._increment
    action[:3] = clipped_ac

    ### robot env step
    action = np.clip(action, self.cont_action_space.low, self.cont_action_space.high)
    for _ in range(num_steps):
      self._set_action(action)
      self.sim.step()
      self._step_callback()
    obs = self._get_obs()

    done = False
    info = {
        'is_success': self._is_success(obs['achieved_goal'], self.goal),
    }
    reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
    #### end robot env step #########
    self.num_step += 1

    if self.eval:
      done = np.allclose(0., reward)
      # info['is_success'] = done
      info = self.add_pertask_success(info, obs['observation'], goal_idx=self.goal_idx)
    else:
      done = False
      # info['is_success'] = np.allclose(0., reward)
      info = self.add_pertask_success(info, obs['observation'], goal_idx=None)

    all_obj_poses = np.split(obs['observation'][5:], self.n)
    z_threshold = 0.5
    for idx, obj_pos in enumerate(all_obj_poses):
      info[f"metric_obj{idx}_above_{z_threshold:.2f}"] = float(obj_pos[2] > z_threshold)

    done = True if self.num_step >= self.max_step else done
    if done: info['TimeLimit.truncated'] = True
    ########## End DemoStackEnv step ############
    return obs, reward, done, info
  def _create_goals(self):
    gripper_offset = np.array([-0.01, 0, 0.008])

    """     g
            2
            1
    """
    final_goal_1 = np.array([1.33193233, 0.74910037, 0.48273329, 0.05 ,  0.05, 1.33193233, 0.74910037, 0.42473329, 1.33193233, 0.74910037, 0.47473329])
    """     g
            1
            2
    """
    temp = np.copy(final_goal_1)
    final_goal_2 = np.copy(final_goal_1)
    final_goal_2[8:11] = final_goal_2[5:8]
    final_goal_2[5:8] = temp[8:11]

    obj0_init_pos = self.initial_qpos['object0:joint'][:3]
    obj1_init_pos = self.initial_qpos['object1:joint'][:3]
    obj0_init_pos[2] = obj1_init_pos[2] = 0.425

    """ g
        0       1
    gripper over first block.
    """
    grip_pos = np.copy(obj0_init_pos) + gripper_offset
    gripper_state = [0.03, 0.03]
    goal_1 = np.concatenate([grip_pos, gripper_state, obj0_init_pos, obj1_init_pos])

    """         g
        0       1
    gripper over 2nd block.
    """
    grip_pos = np.copy(obj1_init_pos) + gripper_offset
    gripper_state = [0.03, 0.03]
    goal_2 = np.concatenate([grip_pos, gripper_state, obj0_init_pos, obj1_init_pos])

    """    g
           0
                 1
    gripper pick first block.
    """
    obj0_lifted_pos = obj0_init_pos + np.array([0, 0, 0.1])
    grip_pos = obj0_lifted_pos + gripper_offset
    gripper_state = [0.0, 0.0]
    goal_3 = np.concatenate([grip_pos, gripper_state, obj0_lifted_pos, obj1_init_pos])

    """    g
           1
       0
    gripper pick second block.
    """
    obj1_lifted_pos = obj1_init_pos + np.array([0, 0, 0.1])
    grip_pos = obj1_lifted_pos + gripper_offset
    gripper_state = [0.0, 0.0]
    goal_4 = np.concatenate([grip_pos, gripper_state, obj0_init_pos, obj1_lifted_pos])


    return np.stack([goal_1, goal_2, goal_3, goal_4, final_goal_1, final_goal_2])


class EasyPickPlaceEnv(PickPlaceEnv):
  # make initialization easier
  def __init__(self,
               max_step=51,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               hard=False,
               distance_threshold=0.,
               n=0.5,
               range_min=0.2,
               range_max=0.45,
               gripper_extra_height=0.16,
               obj_range=0.02,
               target_range=0.05):

    push_1 = [1.34, 0.75 - 0.15, 0.41]
    # push_2 = [1.34, 0.75 + 0.15, 0.41]
    pnp_easy = [1.34, 0.75, 0.52]
    pnp_hard1 = [1.34 - 0.1, 0.75, 0.6]
    pnp_hard2 = [1.34 + 0.1, 0.75, 0.6]

    self.all_goals = np.stack([push_1, pnp_easy, pnp_hard1, pnp_hard2])
    self.goal_idx = -1

    super().__init__(
               max_step=max_step,
               internal_goal=internal_goal,
               external_goal=external_goal,
               mode=mode,
               compute_reward_with_internal=compute_reward_with_internal,
               per_dim_threshold=per_dim_threshold,
               hard=hard,
               distance_threshold=distance_threshold,
               n=n,
               range_min=range_min,
               range_max=range_max,
               gripper_extra_height=gripper_extra_height,
               obj_range=obj_range,
               target_range=target_range)

  def _reset_sim(self):
      self.sim.set_state(self.initial_state)

      # Randomize start position of object.
      if self.has_object:
          object_xpos = self.initial_gripper_xpos[:2]
          # while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
          object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
          object_qpos = self.sim.data.get_joint_qpos('object0:joint')
          assert object_qpos.shape == (7,)
          object_qpos[:2] = object_xpos
          self.sim.data.set_joint_qpos('object0:joint', object_qpos)

      self.sim.forward()
      return True

  def set_goal_idx(self, idx):
    self.goal_idx = idx

  def get_goal_idx(self):
    return self.goal_idx

  def get_goals(self):
    # push left
    # push right
    # pick place to directly above spawn, easy.
    # pick place to somewhere further away.
    return self.all_goals

  def _sample_goal(self):
    return self.all_goals[self.goal_idx]


###########
# Environments with random weight matrix goal projection
# Same as the environments above, but the goal vector returned is now multiplied
# by a (fixed) random vector, initialized when the env is instantiated
##########


class PushRandGoalDistEnv(PushEnv):
  def __init__(self,
               max_step=51,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               compute_external_reward_with_soft_threshold=0,
               seed=123):
    self.done_init = False
    # May be there's a nicer way to pass down the init to parent
    super().__init__(max_step=max_step,
                     internal_goal=internal_goal,
                     external_goal=external_goal,
                     mode=mode,
                     compute_reward_with_internal=compute_reward_with_internal,
                     per_dim_threshold=per_dim_threshold,
                     compute_external_reward_with_soft_threshold=compute_external_reward_with_soft_threshold)
    self.done_init = True
    # Additionally sample a random invertible matrix
    self.seed(seed)

    # Get the size of the goal for this configuration from parent class
    goal_shape = super()._sample_goal().shape

    W = self.np_random.randn(goal_shape[0], goal_shape[0])
    # Check if W is invertible, sample new ones if not
    while not np.isfinite(np.linalg.cond(W)):
      W = self.np_random.randn(goal_shape[0], goal_shape[0])

    self.rand_goal_W = W
    self.rand_goal_W_inv = np.linalg.inv(W)

  def _sample_goal(self):
    goal = super()._sample_goal()

    # Check if has done init yet. If not then just use the original goal space
    if self.done_init:
      # Apply random distillation
      return np.dot(self.rand_goal_W, goal)
    else:
      return goal

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    if self.external_goal == GoalType.OBJ:
      goal = self.goal
    elif self.external_goal == GoalType.OBJSPEED:
      goal = self.goal[:3]
    else:
      goal = self.goal[3:6]

    self.sim.model.site_pos[site_id] = goal - sites_offset[0]
    self.sim.forward()

  def _get_obs(self):
    observation = super()._get_obs()

    if self.done_init:
      # Apply random distillation for the achieved goal.
      # desired_goal is already in the distillation form
      observation['achieved_goal'] = np.dot(self.rand_goal_W, observation['achieved_goal'])

    return observation

  def compute_reward(self, achieved_goal, goal, info):
    # Invert the goals back to original goal space, then use compute reward func from parent

    og_achieved_goal = np.dot(self.rand_goal_W_inv, achieved_goal)
    og_desired_goal = np.dot(self.rand_goal_W_inv, goal)

    reward = super().compute_reward(og_achieved_goal, og_desired_goal, info)

    return reward


class SlideRandGoalDistEnv(SlideEnv):
  def __init__(self,
               max_step=51,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               seed=123):
    self.done_init = False
    super().__init__(max_step=max_step,
                     internal_goal=internal_goal,
                     external_goal=external_goal,
                     mode=mode,
                     compute_reward_with_internal=compute_reward_with_internal,
                     per_dim_threshold=per_dim_threshold)
    self.done_init = True
    # Additionally sample a random invertible matrix
    self.seed(seed)

    # Get the size of the goal for this configuration from parent class
    goal_shape = super()._sample_goal().shape

    W = self.np_random.randn(goal_shape[0], goal_shape[0])
    # Check if W is invertible, sample new ones if not
    while not np.isfinite(np.linalg.cond(W)):
      W = self.np_random.randn(goal_shape[0], goal_shape[0])

    self.rand_goal_W = W
    self.rand_goal_W_inv = np.linalg.inv(W)

  def _sample_goal(self):
    goal = super()._sample_goal()

    # Check if has done init yet. If not then just use the original goal space
    if self.done_init:
      # Apply random distillation
      return np.dot(self.rand_goal_W, goal)
    else:
      return goal

  def _get_obs(self):
    observation = super()._get_obs()

    if self.done_init:
      # Apply random distillation for the achieved goal.
      # desired_goal is already in the distillation form
      observation['achieved_goal'] = np.dot(self.rand_goal_W, observation['achieved_goal'])

    return observation

  def compute_reward(self, achieved_goal, goal, info):
    # Invert the goals back to original goal space, then use compute reward func from parent

    og_achieved_goal = np.dot(self.rand_goal_W_inv, achieved_goal)
    og_desired_goal = np.dot(self.rand_goal_W_inv, goal)

    reward = super().compute_reward(og_achieved_goal, og_desired_goal, info)

    return reward


class PushGoalVisualizer(fetch_env.FetchEnv, EzPickle):
  def __init__(self):
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
    }
    fetch_env.FetchEnv.__init__(self,
                                PUSHXML,
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=0.0,
                                target_in_the_air=False,
                                target_offset=0.0,
                                obj_range=0.15,
                                target_range=0.15,
                                distance_threshold=0.05,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')

    EzPickle.__init__(self)

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
    site_id = self.sim.model.site_name2id('target1')
    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0] + np.array([0., 0., 0.05])

    self.sim.forward()


class PPGoalVisualizer(fetch_env.FetchEnv, EzPickle):
  def __init__(self):
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
    }
    fetch_env.FetchEnv.__init__(self,
                                PPXML,
                                has_object=True,
                                block_gripper=False,
                                n_substeps=20,
                                gripper_extra_height=0.2,
                                target_in_the_air=True,
                                target_offset=0.0,
                                obj_range=0.15,
                                target_range=0.15,
                                distance_threshold=0.05,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')

    EzPickle.__init__(self)

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
    site_id = self.sim.model.site_name2id('target1')
    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0] + np.array([-0.01, 0., 0.008])

    self.sim.forward()


class SlideGoalVisualizer(fetch_env.FetchEnv, EzPickle):
  def __init__(self):
    initial_qpos = {
        'robot0:slide0': 0.05,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'table0:slide0': 0.7,
        'table0:slide1': 0.3,
        'table0:slide2': 0.0,
        'object0:joint': [1.7, 1.1, 0.4, 1., 0., 0., 0.],
    }
    fetch_env.FetchEnv.__init__(self,
                                SLIDEXML,
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=-0.02,
                                target_in_the_air=False,
                                target_offset=np.array([0.4, 0.0, 0.0]),
                                obj_range=0.1,
                                target_range=0.3,
                                distance_threshold=0.05,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')

    EzPickle.__init__(self)

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
    site_id = self.sim.model.site_name2id('target1')
    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]

    self.sim.forward()


class PushLeft(fetch_env.FetchEnv, EzPickle):
  def __init__(self, reward_type='sparse'):
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
    }
    self.max_step = 50
    self.num_step = 0
    fetch_env.FetchEnv.__init__(self,
                                ORIGPUSHXML,
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=0.0,
                                target_in_the_air=False,
                                target_offset=0.0,
                                obj_range=0.15,
                                target_range=0.15,
                                distance_threshold=0.05,
                                initial_qpos=initial_qpos,
                                reward_type=reward_type)
    EzPickle.__init__(self)

  def _env_setup(self, initial_qpos):
    for name, value in initial_qpos.items():
      self.sim.data.set_joint_qpos(name, value)
    utils.reset_mocap_welds(self.sim)
    self.sim.forward()

    delta = np.array([-0.2, 0., 0.])
    # Move end effector into position.
    gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height
                               ]) + delta + self.sim.data.get_site_xpos('robot0:grip')
    gripper_rotation = np.array([1., 0., 1., 0.])
    self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
    self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
    for _ in range(10):
      self.sim.step()

    # Extract information for sampling goals.
    self.initial_gripper_xpos = np.array([1.34182673, 0.74891285, 0.41317183])
    if self.has_object:
      self.height_offset = self.sim.data.get_site_xpos('object0')[2]

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

  def _sample_goal(self):
    if self.has_object:
      if self.np_random.random() < 0.15:
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
      else:
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, 0., size=3)
      goal += self.target_offset
      goal[2] = self.height_offset
      if self.target_in_the_air and self.np_random.uniform() < 0.5:
        goal[2] += self.np_random.uniform(0, 0.45)
    else:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
    return goal.copy()


class PushRight(fetch_env.FetchEnv, EzPickle):
  def __init__(self, reward_type='sparse'):
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
    }
    self.max_step = 50
    self.num_step = 0
    fetch_env.FetchEnv.__init__(self,
                                ORIGPUSHXML,
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=0.0,
                                target_in_the_air=False,
                                target_offset=0.0,
                                obj_range=0.15,
                                target_range=0.15,
                                distance_threshold=0.05,
                                initial_qpos=initial_qpos,
                                reward_type=reward_type)
    EzPickle.__init__(self)

  def _env_setup(self, initial_qpos):
    for name, value in initial_qpos.items():
      self.sim.data.set_joint_qpos(name, value)
    utils.reset_mocap_welds(self.sim)
    self.sim.forward()

    delta = np.array([0.2, 0., 0.])
    # Move end effector into position.
    gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height
                               ]) + delta + self.sim.data.get_site_xpos('robot0:grip')
    gripper_rotation = np.array([1., 0., 1., 0.])
    self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
    self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
    for _ in range(10):
      self.sim.step()

    # Extract information for sampling goals.
    self.initial_gripper_xpos = np.array([1.34182673, 0.74891285, 0.41317183])
    if self.has_object:
      self.height_offset = self.sim.data.get_site_xpos('object0')[2]

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False

    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

  def _sample_goal(self):
    if self.has_object:
      if self.np_random.random() < 0.15:
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
      else:
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(0., self.target_range, size=3)
      goal += self.target_offset
      goal[2] = self.height_offset
      if self.target_in_the_air and self.np_random.uniform() < 0.5:
        goal[2] += self.np_random.uniform(0, 0.45)
    else:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
    return goal.copy()


class DisentangledFetchPushEnv(FetchPushEnv):
  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
    if self.has_object:
        object_pos = self.sim.data.get_site_xpos('object0')
        # rotations
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        # velocities
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
    else:
        object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    achieved_goal = np.squeeze(object_pos.copy())
    obs = np.concatenate([
        grip_pos, gripper_state, grip_velp, gripper_vel,
        object_pos.ravel(), object_rot.ravel(), object_velp.ravel(), object_velr.ravel(),
    ])

    return {
        'observation': obs.copy(),
        'achieved_goal': achieved_goal.copy(),
        'desired_goal': self.goal.copy(),
    }




class DisentangledFetchSlideEnv(fetch_env.FetchEnv, EzPickle):
  def __init__(self, distance_threshold=0.05, reward_type='sparse'):
    initial_qpos = {
        'robot0:slide0': 0.05,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.7, 1.1, 0.41, 1., 0., 0., 0.],
    }
    assert False, "NEEEDS TO BE REVISED"
    fetch_env.FetchEnv.__init__(self,
                                ORIGSLIDEXML,
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=-0.02,
                                target_in_the_air=False,
                                target_offset=np.array([0.4, 0.0, 0.0]),
                                obj_range=0.1,
                                target_range=0.3,
                                distance_threshold=distance_threshold,
                                initial_qpos=initial_qpos,
                                reward_type=reward_type)
    EzPickle.__init__(self)

  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
    if self.has_object:
      object_pos = self.sim.data.get_site_xpos('object0')
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
      # velocities
      object_velp = self.sim.data.get_site_xvelp('object0') * dt
      object_velr = self.sim.data.get_site_xvelr('object0') * dt
      # gripper state
      object_rel_pos = object_pos - grip_pos
      object_velp -= grip_velp
    else:
      object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    if not self.has_object:
      achieved_goal = grip_pos.copy()
    else:
      achieved_goal = np.squeeze(object_pos.copy())

    grip_obs = np.concatenate([
        grip_pos,
        gripper_state,
        grip_velp,
        gripper_vel,
    ])

    obj_obs = np.concatenate([
        object_pos.ravel(),
        object_rot.ravel(),
        object_velp.ravel(),
        object_velr.ravel(),
    ])

    grip_obs_padded = np.concatenate((grip_obs, np.zeros_like(obj_obs)), 0)
    obj_obs_padded = np.concatenate((np.zeros_like(grip_obs), obj_obs), 0)

    return {
        'observation': np.stack((grip_obs_padded, obj_obs_padded), 0),
        'achieved_goal': achieved_goal.copy(),
        'desired_goal': self.goal.copy(),
    }


class DisentangledFetchPickAndPlaceEnv(FetchPickAndPlaceEnv):
  def _get_obs(self):
    assert False, "NEEEDS TO BE REVISED"
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
    if self.has_object:
      object_pos = self.sim.data.get_site_xpos('object0')
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
      # velocities
      object_velp = self.sim.data.get_site_xvelp('object0') * dt
      object_velr = self.sim.data.get_site_xvelr('object0') * dt
      # gripper state
      object_rel_pos = object_pos - grip_pos
      object_velp -= grip_velp
    else:
      object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    if not self.has_object:
      achieved_goal = grip_pos.copy()
    else:
      achieved_goal = np.squeeze(object_pos.copy())

    grip_obs = np.concatenate([
        grip_pos,
        gripper_state,
        grip_velp,
        gripper_vel,
    ])

    obj_obs = np.concatenate([
        object_pos.ravel(),
        object_rot.ravel(),
        object_velp.ravel(),
        object_velr.ravel(),
    ])

    grip_obs_padded = np.concatenate((grip_obs, np.zeros_like(obj_obs)), 0)
    obj_obs_padded = np.concatenate((np.zeros_like(grip_obs), obj_obs), 0)

    return {
        'observation': np.stack((grip_obs_padded, obj_obs_padded), 0),
        'achieved_goal': achieved_goal.copy(),
        'desired_goal': self.goal.copy(),
    }


class SlideNEnv(fetch_env.FetchEnv, EzPickle):
  def __init__(self,
               n=1,
               distance_threshold=0.075,
               **kwargs):
    self.n = n
    self.disentangled_idxs = [np.arange(10)] + [10 + 12*i + np.arange(12) for i in range(n)]
    self.disentangled_goal_idxs = [3*i + np.arange(3) for i in range(n)]
    self.ag_dims = np.concatenate([a[:3] for a in self.disentangled_idxs[1:]])
    if not distance_threshold > 1e-5:
      distance_threshold = 0.075 # default

    initial_qpos = {
        'robot0:slide0': 0.05,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
    }
    for i in range(self.n):
      k = 'object{}:joint'.format(i)
      initial_qpos[k] = INIT_Q_POSES_SLIDE[i]


    fetch_env.FetchEnv.__init__(self,
                                SLIDE_N_XML.replace('#', '{}'.format(n)),
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=0.,
                                target_in_the_air=False,
                                target_offset=np.array([-0.075, 0.0, 0.0]),
                                obj_range=0.15,
                                target_range=0.30,
                                distance_threshold=distance_threshold,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')
    EzPickle.__init__(self)

    self.max_step = 50 + 25 * (n - 1)
    self.num_step = 0


  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

  def _reset_sim(self):
    self.sim.set_state(self.initial_state)

    # Only a little randomize about the start state
    # for i in range(self.n):
    #   object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
    #   object_qpos[:2] += self.np_random.uniform(-0.03, 0.03, size=2)
    #   self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)

    bad_poses = [self.initial_gripper_xpos[:2]]
    # Randomize start positions of pucks.
    for i in range(self.n):
      object_xpos = self.initial_gripper_xpos[:2]
      while min([np.linalg.norm(object_xpos - p) for p in bad_poses]) < 0.08:
          object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
      bad_poses.append(object_xpos)

      object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
      object_qvel = self.sim.data.get_joint_qvel('object{}:joint'.format(i))
      object_qpos[:2] = object_xpos
      object_qpos[2:] = np.array([0.42, 1., 0., 0., 0.])
      self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)
      self.sim.data.set_joint_qvel('object{}:joint'.format(i), np.zeros_like(object_qvel))

    self.sim.forward()
    return True

  def _sample_goal(self):
    first_puck = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)

    goal_xys = [first_puck[:2]]
    for i in range(self.n - 1):
      object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)
      while min([np.linalg.norm(object_xpos - p) for p in goal_xys]) < 0.08:
        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)
      goal_xys.append(object_xpos)

    goals = [np.concatenate((goal, [self.height_offset])) for goal in goal_xys]

    return np.concatenate(goals)

  def _env_setup(self, initial_qpos):
      for name, value in initial_qpos.items():
          self.sim.data.set_joint_qpos(name, value)
      utils.reset_mocap_welds(self.sim)
      self.sim.forward()

      # Move end effector into position.
      gripper_target = np.array([-0.548, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
      gripper_rotation = np.array([1., 0., 1., 0.])
      self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
      self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
      for _ in range(10):
          self.sim.step()

      # Extract information for sampling goals.
      self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
      if self.has_object:
          self.height_offset = self.sim.data.get_site_xpos('object0')[2]

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    info['is_success'] = np.allclose(reward, 0.)
    return obs, reward, done, info

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos)
    goals = np.split(self.goal, self.n)

    for i in range(self.n):
      site_id = self.sim.model.site_name2id('target{}'.format(i))
      self.sim.model.site_pos[site_id] = goals[i] - sites_offset[i]
    self.sim.forward()

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.

    if len(achieved_goal.shape) == 1:
      actual_goals = np.split(goal, self.n)
      achieved_goals = np.split(achieved_goal, self.n)
      success = 1.
    else:
      actual_goals = np.split(goal, self.n, axis=1)
      achieved_goals = np.split(achieved_goal, self.n, axis=1)
      success = np.ones(achieved_goal.shape[0], dtype=np.float32)

    for b, g in zip(achieved_goals, actual_goals):
      d = goal_distance(b, g)
      success *= (d <= self.distance_threshold).astype(np.float32)

    return success - 1.

  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

    obj_feats = []
    obj_poses = []

    for i in range(self.n):
      obj_labl = 'object{}'.format(i)
      object_pos = self.sim.data.get_site_xpos(obj_labl).ravel()
      object_pos[2] = max(object_pos[2], self.height_offset)
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(obj_labl)).ravel()
      # velocities
      object_velp = (self.sim.data.get_site_xvelp(obj_labl) * dt).ravel()
      object_velr = (self.sim.data.get_site_xvelr(obj_labl) * dt).ravel()
      # gripper state
      object_rel_pos = object_pos - grip_pos
      #object_velp -= grip_velp

      obj_feats.append([
        object_pos.ravel(),
        object_rot.ravel(),
        object_velp.ravel(),
        object_velr.ravel(),
      ])
      obj_poses.append(object_pos)

    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    achieved_goal = np.concatenate(obj_poses)

    grip_obs = np.concatenate([
        grip_pos,
        gripper_state,
        grip_velp,
        gripper_vel,
    ])

    obs = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel] + sum(obj_feats, []))

    return {
        'observation': obs,
        'achieved_goal': achieved_goal,
        'desired_goal': self.goal.copy(),
    }


class PushNEnv(fetch_env.FetchEnv, EzPickle):
  def __init__(self,
               n=1,
               distance_threshold=0.05,
               **kwargs):
    self.n = n
    self.disentangled_idxs = [np.arange(10)] + [10 + 12*i + np.arange(12) for i in range(n)]
    self.ag_dims = np.concatenate([a[:3] for a in self.disentangled_idxs[1:]])
    if not distance_threshold > 1e-5:
      distance_threshold = 0.05 # default

    initial_qpos = {
        'robot0:slide0': 0.05,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
    }
    for i in range(self.n):
      k = 'object{}:joint'.format(i)
      initial_qpos[k] = INIT_Q_POSES_SLIDE[i]


    fetch_env.FetchEnv.__init__(self,
                                PUSH_N_XML.replace('#', '{}'.format(n)),
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=0.,
                                target_in_the_air=False,
                                target_offset=np.array([-0.075, 0.0, 0.0]),
                                obj_range=0.15,
                                target_range=0.25,
                                distance_threshold=distance_threshold,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')
    EzPickle.__init__(self)

    self.max_step = 50 + 25 * (n - 1)
    self.num_step = 0


  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

  def _reset_sim(self):
    self.sim.set_state(self.initial_state)

    # Only a little randomize about the start state
    # for i in range(self.n):
    #   object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
    #   object_qpos[:2] += self.np_random.uniform(-0.03, 0.03, size=2)
    #   self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)

    bad_poses = [self.initial_gripper_xpos[:2]]
    # Randomize start positions of pucks.
    for i in range(self.n):
      object_xpos = self.initial_gripper_xpos[:2]
      while min([np.linalg.norm(object_xpos - p) for p in bad_poses]) < 0.08:
          object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
      bad_poses.append(object_xpos)

      object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
      object_qvel = self.sim.data.get_joint_qvel('object{}:joint'.format(i))
      object_qpos[:2] = object_xpos
      object_qpos[2:] = np.array([0.42, 1., 0., 0., 0.])
      self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)
      self.sim.data.set_joint_qvel('object{}:joint'.format(i), np.zeros_like(object_qvel))

    self.sim.forward()
    return True

  def _sample_goal(self):
    first_puck = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)

    goal_xys = [first_puck[:2]]
    for i in range(self.n - 1):
      object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)
      while min([np.linalg.norm(object_xpos - p) for p in goal_xys]) < 0.08:
        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)
      goal_xys.append(object_xpos)

    goals = [np.concatenate((goal, [self.height_offset])) for goal in goal_xys]

    return np.concatenate(goals)

  def _env_setup(self, initial_qpos):
      for name, value in initial_qpos.items():
          self.sim.data.set_joint_qpos(name, value)
      utils.reset_mocap_welds(self.sim)
      self.sim.forward()

      # Move end effector into position.
      gripper_target = np.array([-0.548, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
      gripper_rotation = np.array([1., 0., 1., 0.])
      self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
      self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
      for _ in range(10):
          self.sim.step()

      # Extract information for sampling goals.
      self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
      if self.has_object:
          self.height_offset = self.sim.data.get_site_xpos('object0')[2]

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    info['is_success'] = np.allclose(reward, 0.)
    return obs, reward, done, info

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos)
    goals = np.split(self.goal, self.n)

    for i in range(self.n):
      site_id = self.sim.model.site_name2id('target{}'.format(i))
      self.sim.model.site_pos[site_id] = goals[i] - sites_offset[i]
    self.sim.forward()

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.

    if len(achieved_goal.shape) == 1:
      actual_goals = np.split(goal, self.n)
      achieved_goals = np.split(achieved_goal, self.n)
      success = 1.
    else:
      actual_goals = np.split(goal, self.n, axis=1)
      achieved_goals = np.split(achieved_goal, self.n, axis=1)
      success = np.ones(achieved_goal.shape[0], dtype=np.float32)

    for b, g in zip(achieved_goals, actual_goals):
      d = goal_distance(b, g)
      success *= (d <= self.distance_threshold).astype(np.float32)

    return success - 1.

  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

    obj_feats = []
    obj_poses = []

    for i in range(self.n):
      obj_labl = 'object{}'.format(i)
      object_pos = self.sim.data.get_site_xpos(obj_labl).ravel()
      object_pos[2] = max(object_pos[2], self.height_offset)
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(obj_labl)).ravel()
      # velocities
      object_velp = (self.sim.data.get_site_xvelp(obj_labl) * dt).ravel()
      object_velr = (self.sim.data.get_site_xvelr(obj_labl) * dt).ravel()
      # gripper state
      object_rel_pos = object_pos - grip_pos
      #object_velp -= grip_velp

      obj_feats.append([
        object_pos.ravel(),
        object_rot.ravel(),
        object_velp.ravel(),
        object_velr.ravel(),
      ])
      obj_poses.append(object_pos)

    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    achieved_goal = np.concatenate(obj_poses)

    grip_obs = np.concatenate([
        grip_pos,
        gripper_state,
        grip_velp,
        gripper_vel,
    ])

    obs = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel] + sum(obj_feats, []))

    return {
        'observation': obs,
        'achieved_goal': achieved_goal,
        'desired_goal': self.goal.copy(),
    }
