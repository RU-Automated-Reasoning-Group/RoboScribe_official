from collections import OrderedDict
import gym.spaces as space
import numpy as np
import sapien.core as sapien
import trimesh
from sapien.core import Pose
from scipy.spatial import distance as sdist

from mani_skill2.agents.robots.mobile_panda import MobilePandaSingleArm
from mani_skill2.agents.robots.panda import Panda
from mani_skill2.utils.common import np_random, random_choice, compute_angle_between,flatten_state_dict
from mani_skill2.utils.geometry import angle_distance, transform_points
from mani_skill2.utils.registration import register_env

from mani_skill2.utils.sapien_utils import (
    get_entity_by_name,
    look_at,
    set_articulation_render_material,
    get_pairwise_contact_impulse,
    check_actor_static,
    vectorize_pose,
)
from mani_skill2.envs.pick_and_place.stack_cube import UniformSampler
from .open_cabinet_door_drawer import OpenCabinetEnv
from transforms3d.euler import euler2quat

import pdb

@register_env("OpenCabinetDrawerCube2ArmToppanel-v2", max_episode_steps=400)
class OpenCabinetDrawerCube2ArmToppanelEnv(OpenCabinetEnv):
    place_goal_thresh = 0.025
    pick_min_goal_dist = 0.05
    cube_half_size = np.array([0.02, 0.02, 0.02], np.float32)
    
    DEFAULT_MODEL_JSON = (
        "{PACKAGE_ASSET_DIR}/partnet_mobility/meta/info_cabinet_drawer_train.json"
    )
    
    def __init__(self, task: str = "opendrawer",extract: str="None", solve_cube: int=1, *args,  **kwargs):
        self.task = task
        self.robot_init_qpos_noise=0.02
        self._extract = extract
        self.solve_cube = solve_cube

        super().__init__(*args, **kwargs)
        assert type(self.agent.action_space)==space.Box, type(self.agent.action_space)
        assert type(self.agent2.action_space)==space.Box, type(self.agent2.action_space)
        assert self.agent.action_space.dtype==self.agent2.action_space.dtype
        if self._extract in ["cube"]:
            self.action_space = self.agent2.action_space
        elif self._extract in ["drawer"]:
            # self.action_space = self.agent.action_space
            self.action_space = self.agent2.action_space
        else:
            low = np.concatenate((self.agent.action_space.low, self.agent2.action_space.low))
            high = np.concatenate((self.agent.action_space.high, self.agent2.action_space.high))
            self.action_space = space.Box(low=low, high=high,dtype = self.agent.action_space.dtype)
        
        
    def _set_cabinet_handles(self):
        super()._set_cabinet_handles("prismatic")
    
    def _initialize_task(self):
        """
        reset() ->(last one fn) self.initialize_episode() -> (last one fn) self._initialize_task()
        """
        self._initialize_cabinet()
        self._initialize_robot()
        self._initialize_actors()
        self._set_target_link()
        self._set_joint_physical_parameters()
        
        self._init_env_by_tasktype(self.task)
    
            
    def _load_actors(self):
        super()._load_actors()
        self.cubeA = self._build_cube(self.cube_half_size, color=(1, 0, 0), name="cubeA")
        self.cubeB = self._build_cube(self.cube_half_size, color=(0, 1, 0), name="cubeB")
        self.pick_goal_site = self._build_sphere_site(self.place_goal_thresh, color=(1, 0, 0))
        self.place_goal_site = self._build_sphere_site(self.place_goal_thresh, name="place_goal_site")
        
        # self.cubeFix = self._build_cube(self.cube_half_size, color=(0, 0, 1), name="cubeFix")
        self.cubeFix = self._build_sphere_site(self.cube_half_size[0], color=(1, 1, 1), name="cubeFix")
        
        self.cubeC = self._build_cube(self.cube_half_size, color=(0, 0, 1), name="cubeC")

        self.obs_cubeA = self.cubeA
        self.obs_cubeB = self.cubeB
        self.obs_cubeC = self.cubeC

        
    def _initialize_actors(self):
        xy = self._episode_rng.uniform(-0.1, 0.1, [2])
        # region = [[-0.1, -0.2],[0.1, 0.2]]
        region = [[-0.1, -0.1],[0.1, 0.1]]
        # region = [[-0.05, -0.05],[0.05, 0.05]]
        sampler = UniformSampler(region, self._episode_rng)
        radius = np.linalg.norm(self.cube_half_size[:2]) + 0.02
        cubeA_xy = xy + sampler.sample(radius, 200, verbose=True)
        cubeB_xy = xy + sampler.sample(radius, 200, verbose=False)
        cubeA_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        cubeB_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        # z = self.box_half_size[2]
        z = 0.79
        cubeA_pose = sapien.Pose([cubeA_xy[0], cubeA_xy[1], z], cubeA_quat)
        cubeB_pose = sapien.Pose([cubeB_xy[0], cubeB_xy[1], z], cubeB_quat)

        self.cubeA.set_pose(cubeA_pose)
        self.cubeB.set_pose(cubeB_pose)
        
        
        cubeFix_xy = xy + sampler.sample(radius, 100, verbose=False, append=False)
        cubeFix_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        cubeFix_pose = sapien.Pose([cubeFix_xy[0], cubeFix_xy[1], z], cubeFix_quat)
        self.cubeFix.set_pose(cubeFix_pose)
        
        cubeC_xy = xy + sampler.sample(radius, 200, verbose=False)
        cubeC_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        cubeC_pose = sapien.Pose([cubeC_xy[0], cubeC_xy[1], z], cubeC_quat)
        self.cubeC.set_pose(cubeC_pose)

        self.obs_cubeA = self.cubeA
        self.obs_cubeB = self.cubeB
        self.obs_cubeC = self.cubeC

    def _init_env_by_tasktype(self, task):
        """
        For different task, we should set different robot arm pos, cabinet drawer openness and (camera position) 
        """
        # set arm agent1 (far)
        center = np.array([0, 0.8])
        # center = np.array([10, 10])
        dist = self._episode_rng.uniform(1.6, 1.8)
        theta = self._episode_rng.uniform(0.9 * np.pi, 1.1 * np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)])
        xy = center + direction * dist
        # Base orientation
        noise_ori = self._episode_rng.uniform(-0.05 * np.pi, 0.05 * np.pi)
        ori = (theta - np.pi) + noise_ori
        h = 1e-4
        arm_qpos = np.array([0, 0, 0, -1.5, 0, 3, 0.78, 0.02, 0.02])
        qpos_1 = np.hstack([xy, ori, h, arm_qpos])
        self.agent.reset(qpos_1)
        
        # set arm agent2 (near)
        qpos_2 = np.array(
                [-np.pi/2, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04])
        qpos_2[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos_2) - 2
            )
        self.agent2.reset(qpos_2)
        self.agent2.robot.set_pose(Pose([0, 0.615, 0.79+0.17]))
        
        # pick_goal_site
        obj_pos = self.cubeA.pose.p
        max_trials=100
        # Sample a goal position far enough from the object
        for i in range(max_trials):
            goal_xy = self._episode_rng.uniform(-0.1, 0.1, [2])
            goal_z = self._episode_rng.uniform(0, 0.5) + obj_pos[2]
            goal_pos = np.hstack([goal_xy, goal_z])
            if np.linalg.norm(goal_pos - obj_pos) > self.pick_min_goal_dist:
                break

        self.goal_pos = goal_pos
        self.pick_goal_site.set_pose(Pose(self.goal_pos))
        
        qmin, qmax = self.target_joint.get_limits()[0]
        canbinet_pos = [0, 0, 0.0, qmin]
        self.open_target_qpos = qmin + (qmax - qmin) * 0.9
        self.close_target_qpos = qmin + (qmax - qmin) * 0.1
        
        # set drawer
        if "pickplace" in self.task :
            rd = self._episode_rng.uniform(0.9, 1)
            canbinet_pos[3] += rd*(qmax - qmin)
            self.cabinet.set_qpos(canbinet_pos)
        else:
            # rd = self._episode_rng.uniform(0, 1)
            # canbinet_pos[3] += rd*(qmax - qmin)
            self.cabinet.set_qpos(canbinet_pos)
        # canbinet_pos[3] += qmax - qmin
        # self.cabinet.set_qpos(canbinet_pos)
        
        # place_goal_site
        self.place_goal_site.set_pose(Pose(self.target_link.pose.p + [-0.25, 0.0, 0.015]))
        
    def reconfigure(self):
        """Reconfigure the simulation scene instance.
        This function should clear the previous scene, and create a new one.
        """
        self._clear()
        self._setup_scene()
        self._load_agent()
        self._load_actors()
        self._load_articulations()
        self._setup_cameras()
        self._setup_lighting()
        
        if self._viewer is not None:
            self._setup_viewer()

        # Cache actors and articulations
        self._actors = self.get_actors()
        self._articulations = self.get_articulations()

        self._load_background()
        
    def _load_agent(self):
        self.agent = MobilePandaSingleArm(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )
        self.agent2 = Panda(
            self._scene, self._control_freq, "pd_ee_delta_pose", config=Panda.get_default_config()
        )
        
        self.tcp: sapien.Link = get_entity_by_name(
            self.agent2.robot.get_links(), self.agent2.config.ee_link_name
        )

        
    def _register_render_cameras(self):
        cam_cfg = super()._register_render_cameras()
        # if self.task in ["pickcube", ]:
        #     # for cube on the cabinet side shelf
        #     pose = look_at([0, 0, 1.3], [-0.3, 0.37, 0.765])
        #     cam_cfg.p = pose.p
        #     cam_cfg.q = pose.q
        #     # for cube on the floor
        #     # pose = look_at([0, 2, 1], [-1, 0, 0.02])
        #     # cam_cfg.p = pose.p
        #     # cam_cfg.q = pose.q
        cam_cfg.p = [-1.5, 0.3, 2]
        cam_cfg.q = [0.9238795, 0, 0.3826834, 0]
        return cam_cfg
        
    def _build_cube(
        self,
        half_size,
        color=(1, 0, 0),
        name="cube",
        static=False,
        render_material: sapien.RenderMaterial = None,
    ):
        if render_material is None:
            render_material = self._renderer.create_material()
            render_material.set_base_color(np.hstack([color, 1.0]))

        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=render_material)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)
        
    def _build_sphere_site(self, radius, color=(0, 1, 0), name="pick_goal_site"):
        """Build a sphere site (visual only). Used to indicate goal position."""
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, color=color)
        sphere = builder.build_static(name)
        # NOTE(jigu): Must hide after creation to avoid pollute observations!
        sphere.hide_visual()
        return sphere
    
    def render(self, mode="human"):
        if mode in ["human", "rgb_array"]:
            self.pick_goal_site.unhide_visual()
            # self.place_goal_site.unhide_visual()
            self.cubeFix.unhide_visual()
            ret = super().render(mode=mode)
            self.pick_goal_site.hide_visual()
            # self.place_goal_site.hide_visual()
            self.cubeFix.unhide_visual()
        else:
            ret = super().render(mode=mode)
        return ret
    
    def get_scene(self):
        return self._scene
    
    def check_grasp_handle(self):
        ee_coords = self.agent.get_ee_coords_sample()
        check_handle_pcd = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)
        check_disp_ee_to_handle = sdist.cdist(ee_coords.reshape(-1, 3), check_handle_pcd)
        check_dist_ee_to_handle = check_disp_ee_to_handle.reshape(2, -1).min(-1)
        
        check_ee_center_pos = sdist.cdist(np.expand_dims(self.agent.get_ee_coords().mean(0), 0), transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)).min()
        if check_ee_center_pos < 0.02 and check_dist_ee_to_handle.sum() < 0.02:
            return True
        else:
            return False

    def _check_cubeA_on_cubeB(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            np.linalg.norm(offset[:2]) <= np.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = np.abs(offset[2] - self.cube_half_size[2] * 2) <= 0.005
        return bool(xy_flag and z_flag)
    
    def check_cubeA_on_cubeB(self):
        return self._check_cubeA_on_cubeB()
    
    def check_cube_in_drawer(self, cube):
        # behind_left: [-0.06, 0.14, 0.01]
        # behind_right: [-0.06, -0.14, 0.015]
        # front_left: [-0.44, 0.14, 0.015]
        # front_right: [-0.44, -0.14, 0.015]
        # height [0.67, 0.69]
        threshold = np.array([0.19,0.14,0.03])
        center = self.target_link.pose.p + [-0.25, 0.0, 0.01]
        abs_dis = np.abs(cube.pose.p-center)
        if (abs_dis<threshold).all():
            return True
        return False
        
    def set_extract(self, extract):
        self._extract = extract
        if self._extract in ["cube"]:
            self.action_space = self.agent2.action_space
        elif self._extract in ["drawer"]:
            # self.action_space = self.agent.action_space
            self.action_space = self.agent2.action_space
        else:
            low = np.concatenate((self.agent.action_space.low, self.agent2.action_space.low))
            high = np.concatenate((self.agent.action_space.high, self.agent2.action_space.high))
            self.action_space = space.Box(low=low, high=high,dtype = self.agent.action_space.dtype)
        
    def get_extract(self,):
        return self._extract
        
    def _get_obs_agent(self, join=False):
        if self._extract in ["cube"]:
            agent2_obs = self.agent2.get_proprioception()
            agent2_obs["base_pose"] = vectorize_pose(self.agent2.robot.pose)
            return agent2_obs
        elif self._extract in ["drawer"]:
            # return self.agent.get_proprioception()
            if join:
                agent2_obs = self.agent2.get_proprioception()
                agent2_obs["base_pose"] = vectorize_pose(self.agent2.robot.pose)
                return agent2_obs
            else:
                return self.agent2.get_proprioception()
        else:
            agent_obs=self.agent.get_proprioception()
            agent2_obs = self.agent2.get_proprioception()
            agent2_obs["base_pose"] = vectorize_pose(self.agent2.robot.pose)
            return OrderedDict(
                agent=agent_obs,
                agent2=agent2_obs,
            )
    
    def _get_obs_extra(self) -> OrderedDict:
        obs = super()._get_obs_extra()
        if self._extract in ["cube"]:
            obs = OrderedDict(
                tcp_pose=vectorize_pose(self.tcp.pose),
                )
            if self._obs_mode in ["state", "state_dict"]:
                obs.update(
                    cubeA_pose=vectorize_pose(self.cubeA.pose),
                    cubeFix_pose=vectorize_pose(self.cubeFix.pose),
                    pick_goal_pos=self.pick_goal_site.pose.p,
                    place_goal_pos=self.place_goal_site.pose.p,
                    tcp_to_cubeA_pos=self.cubeA.pose.p - self.tcp.pose.p,
                    tcp_to_cubeFix_pos=self.cubeFix.pose.p - self.tcp.pose.p,
                    tcp_to_pick_goal_pos=self.pick_goal_site.pose.p - self.tcp.pose.p,
                    tcp_to_place_goal_pos=self.place_goal_site.pose.p - self.tcp.pose.p,
                    cubeA_to_pick_goal_pos=self.cubeA.pose.p - self.tcp.pose.p,
                    cubeA_to_place_goal_pos=self.cubeA.pose.p - self.tcp.pose.p,
                    # cubeA_to_place_goal_pos=self.cubeA.pose.p - self.place_goal_site.pose.p,
                    cubeA_to_cubeFix_pos=self.cubeFix.pose.p - self.cubeA.pose.p,
                    cubeA_in_drawer=1 if self.check_cube_in_drawer(self.cubeA) else 0,
                )
        elif self._extract in ["drawer"]:
            pass      
        else:
            obs = super()._get_obs_extra()
            if self._obs_mode in ["state", "state_dict"]:
                obs.update(
                    tcp_pose=vectorize_pose(self.tcp.pose),
                    cubeA_pose=vectorize_pose(self.cubeA.pose),
                    cubeFix_pose=vectorize_pose(self.cubeFix.pose),
                    pick_goal_pos=self.pick_goal_site.pose.p,
                    place_goal_pos=self.place_goal_site.pose.p,
                    tcp_to_cubeA_pos=self.cubeA.pose.p - self.tcp.pose.p,
                    tcp_to_cubeFix_pos=self.cubeFix.pose.p - self.tcp.pose.p,
                    tcp_to_pick_goal_pos=self.pick_goal_site.pose.p - self.tcp.pose.p,
                    tcp_to_place_goal_pos=self.place_goal_site.pose.p - self.tcp.pose.p,
                    cubeA_to_pick_goal_pos=self.cubeA.pose.p - self.tcp.pose.p,
                    cubeA_to_place_goal_pos=self.cubeA.pose.p - self.tcp.pose.p,
                    # cubeA_to_place_goal_pos=self.cubeA.pose.p - self.place_goal_site.pose.p,
                    cubeA_to_cubeFix_pos=self.cubeFix.pose.p - self.cubeA.pose.p,
                )

        return obs
    
    def get_obs_2_cubes(self, join=False):
        state_dict = OrderedDict(
            agent=self._get_obs_agent(join),
            extra=self._get_obs_extra_2(join),
        )

        flatten_obs = flatten_state_dict(state_dict)
        if join:
            add_1 = self.link_qpos
            add_2 = self.target_qpos
            add_3 = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd).mean(axis=0)
            flatten_obs = np.concatenate([flatten_obs, np.array([add_1]), np.array([add_2]), add_3], axis=0)

        return flatten_obs


    def _get_obs_extra_2(self, join=False) -> OrderedDict:
        obs = super()._get_obs_extra()
        if self._extract in ["cube"] and not join:
            obs = OrderedDict(
                tcp_pose=vectorize_pose(self.tcp.pose),
                )
            if self._obs_mode in ["state", "state_dict"]:
                obs.update(
                    # place goal
                    place_goal_pos=self.place_goal_site.pose.p,
                    tcp_to_place_goal_pos=self.place_goal_site.pose.p - self.tcp.pose.p,
                    # cube A
                    cubeA_pose=vectorize_pose(self.obs_cubeA.pose),
                    tcp_to_cubeA_pos=self.obs_cubeA.pose.p - self.tcp.pose.p,
                    cubeA_to_place_goal_pos=self.obs_cubeA.pose.p - self.place_goal_site.pose.p,
                    cubeA_in_drawer=1 if self.check_cube_in_drawer(self.obs_cubeA) else 0,
                )
                if self.solve_cube > 1:
                    obs.update(
                        # cube B
                        cubeB_pose=vectorize_pose(self.obs_cubeB.pose),
                        tcp_to_cubeB_pos=self.obs_cubeB.pose.p - self.tcp.pose.p,
                        cubeB_to_place_goal_pos=self.obs_cubeB.pose.p - self.place_goal_site.pose.p,
                        cubeB_in_drawer=1 if self.check_cube_in_drawer(self.obs_cubeB) else 0,
                    )
                    if self.solve_cube > 2:
                        obs.update(
                            # cube C
                            cubeC_pose=vectorize_pose(self.obs_cubeC.pose),
                            tcp_to_cubeC_pos=self.obs_cubeC.pose.p - self.tcp.pose.p,
                            cubeC_to_place_goal_pos=self.obs_cubeC.pose.p - self.place_goal_site.pose.p,
                            cubeC_in_drawer=1 if self.check_cube_in_drawer(self.obs_cubeC) else 0,
                        )
        elif self._extract in ["drawer"] and not join:
            pass
        else:
            obs = super()._get_obs_extra()
            if self._obs_mode in ["state", "state_dict"]:
                obs.update(
                    # place goal
                    tcp_pose=vectorize_pose(self.tcp.pose),
                    place_goal_pos=self.place_goal_site.pose.p,
                    tcp_to_place_goal_pos=self.place_goal_site.pose.p - self.tcp.pose.p,
                    # cubeA
                    cubeA_pose=vectorize_pose(self.obs_cubeA.pose),
                    tcp_to_cubeA_pos=self.obs_cubeA.pose.p - self.tcp.pose.p,
                    cubeA_to_place_goal_pos=self.obs_cubeA.pose.p - self.place_goal_site.pose.p,
                    cubeA_in_drawer=1 if self.check_cube_in_drawer(self.obs_cubeA) else 0,
                )
                if self.solve_cube > 1:
                    obs.update(
                        # cube B
                        cubeB_pose=vectorize_pose(self.obs_cubeB.pose),
                        tcp_to_cubeB_pos=self.obs_cubeB.pose.p - self.tcp.pose.p,
                        cubeB_to_place_goal_pos=self.obs_cubeB.pose.p - self.place_goal_site.pose.p,
                        cubeB_in_drawer=1 if self.check_cube_in_drawer(self.obs_cubeB) else 0,
                    )
                    if self.solve_cube > 2:
                        obs.update(
                            # cube C
                            cubeC_pose=vectorize_pose(self.obs_cubeC.pose),
                            tcp_to_cubeC_pos=self.obs_cubeC.pose.p - self.tcp.pose.p,
                            cubeC_to_place_goal_pos=self.obs_cubeC.pose.p - self.place_goal_site.pose.p,
                            cubeC_in_drawer=1 if self.check_cube_in_drawer(self.obs_cubeC) else 0,
                        )

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.place_goal_site.set_pose(Pose(self.target_link.pose.p + [-0.25, 0.0, 0.01]))
        return obs, reward, done, info
    
    def step_action(self, action):
        agent1_action_dim = self.agent.controller.action_space.shape
        agent2_action_dim = self.agent2.controller.action_space.shape
        agents_action_dim = (agent1_action_dim[0] + agent2_action_dim[0],)
        
        if self._extract in ["cube"]:
            assert action.shape == agent2_action_dim, (action.shape, agent2_action_dim)
            if action is None:  # simulation without action
                pass
            elif isinstance(action, np.ndarray):
                self.agent.set_action(np.zeros(agent1_action_dim[0]))
                self.agent2.set_action(action)
            else:
                raise TypeError(type(action))
        elif self._extract in ["drawer"]:
            # assert action.shape == agent1_action_dim, (action.shape, agent1_action_dim)
            assert action.shape == agent2_action_dim, (action.shape, agent2_action_dim)
            if action is None:  # simulation without action
                pass
            elif isinstance(action, np.ndarray):
                # self.agent.set_action(action)
                # self.agent2.set_action(np.zeros(agent2_action_dim[0]))
                self.agent.set_action(np.zeros(agent1_action_dim[0]))
                self.agent2.set_action(action)
            else:
                raise TypeError(type(action))
        else:
            assert action.shape == agents_action_dim, (action.shape, agents_action_dim)
            if action is None:  # simulation without action
                pass
            elif isinstance(action, np.ndarray):
                self.agent.set_action(action[:agent1_action_dim[0]])
                self.agent2.set_action(action[agent1_action_dim[0]:])
            else:
                raise TypeError(type(action))
            
        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            self.agent.before_simulation_step()
            self.agent2.before_simulation_step()
            self._scene.step()
            self._after_simulation_step()
        
    def reassign_cubeA(self, agent_only=False):
        # set arm agent2 (near)
        qpos_2 = np.array(
                [-np.pi/2, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04])
        qpos_2[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos_2) - 2
            )
        self.agent2.reset(qpos_2)
        self.agent2.robot.set_pose(Pose([0, 0.615, 0.79+0.17]))
        
        if not agent_only:
            tmp = self.cubeA
            self.cubeA = self.cubeB
            self.cubeB = self.cubeC
            self.cubeC = tmp

        # tmp = self.cubeA
        # self.cubeA = self.cubeB
        # self.cubeB = tmp     
    
        