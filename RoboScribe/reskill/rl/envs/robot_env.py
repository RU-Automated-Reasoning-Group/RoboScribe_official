import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding
from mujoco_py import GlfwContext
import pdb


try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )


# class RobotEnv(gym.GoalEnv):
class RobotEnv(gym.Env):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps, seed):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File {} does not exist".format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        # self.physics = mujoco.Physics.from_xml_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.seed_num = seed
        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        # print(self.initial_state)
        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.obs = obs
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
                force_sensor=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
            )
        )

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        # if seed is None:
        #     print("seed input is None")
        # else:
        #     print("seed input is ", seed)
        self.np_random, seed = seeding.np_random(seed)
        # print("seed output is ", seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        self.obs = obs

        done = False

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        # info = {}
        reward = self.compute_reward(
            obs["achieved_goal"],
            self.goal,
            obs["force_sensor"][2],
            obs["observation"][:3],
            obs["observation"][3:5],
            obs,
            info,
        )
        # print("Sim State", self.sim.get_state())
        # print("box position", self.obs["achieved_goal"])
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        self.obs = obs
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None

    def render(self, mode="human", width=3000, height=2000):
        self._render_callback()
        if mode == "rgb_array":
            self._get_viewer(mode).render(width, height)
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # self.sim.render(width=300, height=200, camera_name='external_camera_0', depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]

        elif mode == "human":
            self._get_viewer(mode).render()

    # def _get_viewer(self):
    #     if self.viewer is None:
    #         self.viewer = mujoco_py.MjViewer(self.sim)
    #         self._viewer_setup()
    #     return self.viewer

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        # self.viewer.scn.maxgeom = 100000
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation."""
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation."""
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
