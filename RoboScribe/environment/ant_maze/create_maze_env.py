# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

# from envs.sibrivalry.ant_maze.ant_maze_env import AntMazeEnv, AntMazeEnvFull, AntMazeEnvFullDownscale, A1MazeEnvFullDownscale
from environment.ant_maze.ant_maze_env import AntMazeEnv, AntMazeEnvFull, AntMazeEnvFullDownscale, A1MazeEnvFullDownscale


def create_maze_env(env_name=None, top_down_view=False, observation_space=None):
  n_bins = 0
  manual_collision = False
  if env_name.startswith('Ego'):
    n_bins = 8
    env_name = env_name[3:]
  elif env_name.startswith('Ant') and env_name.endswith('Full'):
    cls = AntMazeEnvFull
    env_name = env_name[3:]
    maze_size_scaling = 8
  elif env_name.startswith('A1') and env_name.endswith('Downscale'):
    cls = A1MazeEnvFullDownscale
    env_name = env_name[2:10]
    maze_size_scaling = 1
  elif env_name.startswith('Ant') and env_name.endswith('Downscale'):
    cls = AntMazeEnvFullDownscale
    if 'HardMaze' in env_name:
      env_name = env_name[3:15]
    else:
      env_name = env_name[3:11]
    maze_size_scaling = 2
  elif env_name.startswith('Ant') and not env_name.endswith('Full'):
    cls = AntMazeEnv
    env_name = env_name[3:]
    maze_size_scaling = 8
  else:
    assert False, 'unknown env %s' % env_name

  observe_blocks = False
  put_spin_near_agent = False
  if env_name == 'Maze' or env_name == 'MazeFull':
    maze_id = 'Maze'
  elif env_name == 'HardMazeFull':
    maze_id = 'HardMaze'
  elif env_name == 'Push':
    maze_id = 'Push'
    observe_blocks = True
  elif env_name == 'Fall':
    maze_id = 'Fall'
    observe_blocks = True
  elif env_name == 'Block':
    maze_id = 'Block'
    put_spin_near_agent = True
    observe_blocks = True
  elif env_name == 'BlockMaze':
    maze_id = 'BlockMaze'
    put_spin_near_agent = True
    observe_blocks = True
  else:
    raise ValueError('Unknown maze environment %s' % env_name)

  gym_mujoco_kwargs = {
      'maze_id': maze_id,
      'n_bins': n_bins,
      'observe_blocks': observe_blocks,
      'put_spin_near_agent': put_spin_near_agent,
      'top_down_view': top_down_view,
      'manual_collision': manual_collision,
      'maze_size_scaling': maze_size_scaling,
      'observation_space': observation_space
  }
  gym_env = cls(**gym_mujoco_kwargs)
  gym_env.reset()
  return gym_env