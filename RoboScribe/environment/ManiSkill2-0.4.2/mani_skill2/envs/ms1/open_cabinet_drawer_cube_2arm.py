from collections import OrderedDict

import numpy as np
import sapien.core as sapien
import trimesh
from sapien.core import Pose
from scipy.spatial import distance as sdist

from mani_skill2.agents.robots.mobile_panda import MobilePandaSingleArm
from mani_skill2.utils.common import np_random, random_choice, compute_angle_between
from mani_skill2.utils.geometry import angle_distance, transform_points
from mani_skill2.utils.registration import register_env

from mani_skill2.utils.sapien_utils import (
    get_entity_by_name,
    get_pairwise_contact_impulse,
    vectorize_pose,
)
from mani_skill2.utils.sapien_utils import (
    get_entity_by_name,
    look_at,
    set_articulation_render_material,
    vectorize_pose,
)

from .open_cabinet_door_drawer import OpenCabinetEnv

@register_env("OpenCabinetDrawerCube2Arm-v1", max_episode_steps=400)
class OpenCabinetDrawerCube2ArmEnv(OpenCabinetEnv):
    place_goal_thresh = 0.025
    cube_half_size = np.array([0.02, 0.02, 0.02], np.float32)
    
    DEFAULT_MODEL_JSON = (
        "{PACKAGE_ASSET_DIR}/partnet_mobility/meta/info_cabinet_drawer_train.json"
    )
    _task_type = ["opendrawer","closedrawer","pickcube"]
    
    def __init__(self, task: str = "opendrawer", *args,  **kwargs):
        assert task in self._task_type
        self._task = task
        super().__init__(*args, **kwargs)
        
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
        
        self._init_env_by_tasktype(self._task)
    
    def _initialize_actors(self):
        # cube on the side shelf
        cube_xyz = np.hstack([[0.01, 0.37], 0.765])
        
        cube_noise = self._episode_rng.uniform(-0.01, 0.01)
        cube_xyz[0] += cube_noise
        q = [1, 0, 0, 0]
        self.obj.set_pose(Pose(cube_xyz, q))
    
    def _init_env_by_tasktype(self, task):
        """
        For different task, we should set different robot arm pos, cabinet drawer openness and (camera position) 
        """
        # set arm agent1 (far)
        center = np.array([0, 0.8])
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
        self.agent1.reset(qpos_1)
        
        # set arm agent2 (close)
        noise_ori = self._episode_rng.uniform(-0.05 * np.pi, 0.05 * np.pi)
        xy = np.array([-0.6, 1.2])
        h = 0.5
        ori = np.array([0.5]) + noise_ori
        arm_qpos = np.array([-1.5, -1.5, 0, -1.5, 0, 1.5, 1.2, 0.04, 0.04])
        qpos_2 = np.hstack([xy, ori, h, arm_qpos])
        self.agent2.reset(qpos_2)
        
        self.agent = self.agent1
        self.inactive_agent = self.agent2
        
        if task in ["opendrawer"]:
            # set target_qpos
            qmin, qmax = self.target_joint.get_limits()[0]
            self.target_qpos = qmin + (qmax - qmin) * 0.9
            
            # set drawer
            # canbinet_pos = [0, 0, 0.0, qmin]
            # rd = self._episode_rng.uniform(0, 0.5)
            # canbinet_pos[3] += rd*(qmax - qmin)
            canbinet_pos = [0, 0, 0.0, 0.0]
            self.cabinet.set_qpos(canbinet_pos)
            
            #set cube goal
            self.obj_goal_site.set_pose(Pose(self.target_link.pose.p + [-0.25, 0, 0.0]))
        elif task in ["closedrawer"]:
            # set target_qpos
            qmin, qmax = self.target_joint.get_limits()[0]
            self.target_qpos = qmin + (qmax - qmin) * 0.1
            
            # set drawer
            canbinet_pos = [0, 0, 0.0, qmin]
            rd = self._episode_rng.uniform(0.5, 1)
            canbinet_pos[3] += rd * (qmax - qmin)
            self.cabinet.set_qpos(canbinet_pos)
            
            self.obj_goal_site.set_pose(Pose(self.target_link.pose.p + [-0.25, 0, 0.0]))

        elif task in ["pickcube"]:
            # set drawer
            # qmin, qmax = self.target_joint.get_limits()[0]
            # # self.target_qpos = qmin + (qmax - qmin) * 0.1
            # canbinet_pos = [0, 0, 0.0, qmin]
            # rd = self._episode_rng.uniform(0, 1)
            # canbinet_pos[3] += rd * (qmax - qmin)
            canbinet_pos = [0, 0, 0.0, 0.39]
            self.cabinet.set_qpos(canbinet_pos)
            
            rd = self._episode_rng.uniform(-0.05, 0.05, [2])
            # goal_center = np.array([-0.4 + rd[0],  0.2 + rd[1],  0.7])
            goal_center = np.array([-0.4,  0.2,  0.7])
            self.obj_goal_site.set_pose(Pose(goal_center))
        
    def set_active_agent(self, task):
        if task != self._task:
            self._task = task
            self._init_env_by_tasktype(task)
            if task in ['pickcube']:
                self.agent = self.agent2
                self.inactive_agent = self.agent1
            elif task in ['opendrawer','closedrawer']:
                self.agent = self.agent1
                self.inactive_agent = self.agent2
            

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
        self.agent = self.agent1 = MobilePandaSingleArm(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )
        self.agent2 = MobilePandaSingleArm(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )
        
    def _load_actors(self):
        super()._load_actors()
        self.obj = self._build_cube(self.cube_half_size)
        self.obj_goal_site = self._build_sphere_site(self.place_goal_thresh)
        
    def _register_render_cameras(self):
        cam_cfg = super()._register_render_cameras()
        if self._task in ["pickcube", ]:
            # for cube on the cabinet side shelf
            pose = look_at([0, 0, 1.3], [-0.3, 0.37, 0.765])
            cam_cfg.p = pose.p
            cam_cfg.q = pose.q
            # for cube on the floor
            # pose = look_at([0, 2, 1], [-1, 0, 0.02])
            # cam_cfg.p = pose.p
            # cam_cfg.q = pose.q
        else:
            # # for open/close drawer
            # cam_cfg.p = [-3, 0.3, 2]
            # cam_cfg.q = [0.9238795, 0, 0.3826834, 0]
            # for open/close drawer
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
        
    def _build_sphere_site(self, radius, color=(0, 1, 0), name="obj_goal_site"):
        """Build a sphere site (visual only). Used to indicate goal position."""
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, color=color)
        sphere = builder.build_static(name)
        # NOTE(jigu): Must hide after creation to avoid pollute observations!
        sphere.hide_visual()
        return sphere
    
    def render(self, mode="human"):
        if mode in ["human", "rgb_array"]:
            # self.obj_goal_site.unhide_visual()
            ret = super().render(mode=mode)
            # self.obj_goal_site.hide_visual()
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

    
    def _get_obs_agent(self):
        return self.agent.get_proprioception()
    
    def _get_obs_extra(self) -> OrderedDict:
        obs = super()._get_obs_extra()
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=vectorize_pose(self.obj.pose),
                goal_pos=self.obj_goal_site.pose.p,
            )
        return obs
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info
    
    def step_action(self, action):
        if action is None:  # simulation without action
            pass
        elif isinstance(action, np.ndarray):
            self.agent.set_action(action)
        elif isinstance(action, dict):
            if action["control_mode"] != self.agent.control_mode:
                self.agent.set_control_mode(action["control_mode"])
            self.agent.set_action(action["action"])
        else:
            raise TypeError(type(action))
        
        # keep unused agent arm stable
        self.inactive_agent.set_action(np.array([0,0,0,0,0,0,0,0,0,0,0]))
        
        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            self.agent.before_simulation_step()
            self.inactive_agent.before_simulation_step()
            self._scene.step()
            self._after_simulation_step()
            
    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.agent.scene.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.agent.finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, self.agent.finger2_link, actor)
        print(limpulse)
        print(rimpulse)
        # direction to open the gripper
        ldirection = self.agent.finger1_link.pose.to_transformation_matrix()[:3, 1]
        rdirection = -self.agent.finger2_link.pose.to_transformation_matrix()[:3, 1]

        # angle between impulse and open direction
        langle = compute_angle_between(ldirection, limpulse)
        rangle = compute_angle_between(rdirection, rimpulse)

        lflag = (
            np.linalg.norm(limpulse) >= min_impulse and np.rad2deg(langle) <= max_angle
        )
        rflag = (
            np.linalg.norm(rimpulse) >= min_impulse and np.rad2deg(rangle) <= max_angle
        )

        return all([lflag, rflag])
        
    
    

        