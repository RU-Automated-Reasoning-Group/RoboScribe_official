# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# from envs.sibrivalry.ant_maze.maze_env import MazeEnv, MazeEnvFull
# from envs.sibrivalry.ant_maze.ant import AntEnv
# from envs.sibrivalry.ant_maze.a1 import A1Env
from environment.ant_maze.maze_env import MazeEnv, MazeEnvFull
from environment.ant_maze.ant import AntEnv
from environment.ant_maze.a1 import A1Env

import pdb

class AntMazeEnv(MazeEnv):
    MODEL_CLASS = AntEnv

class AntMazeEnvFull(MazeEnvFull):
    MODEL_CLASS = AntEnv
    def render(self):
        if self.render_mode == "rgb_array":
            return self.wrapped_env.sim.render(height=100, width=100, camera_name="external_camera_0")[::-1]
        else:
            return self.wrapped_env.render(self.render_mode, height=100, width=100)

class AntMazeEnvFullDownscale(MazeEnvFull):
    MODEL_CLASS = AntEnv
    def render(self):
        if self.render_mode == "rgb_array":
            return self.wrapped_env.sim.render(height=100, width=100, camera_name="external_camera_0")[::-1]
        else:
            return self.wrapped_env.render(self.render_mode, height=100, width=100)

class A1MazeEnvFullDownscale(MazeEnvFull):
    MODEL_CLASS = A1Env
    def render(self):
        if self.render_mode == "rgb_array":
            return self.wrapped_env.sim.render(height=150, width=150, camera_name="external_camera_0")[::-1]
        else:
            return self.wrapped_env.render(self.render_mode, height=150, width=150)