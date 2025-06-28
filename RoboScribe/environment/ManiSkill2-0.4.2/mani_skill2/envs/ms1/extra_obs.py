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

def _get_obs_extra_2(self) -> OrderedDict:
    obs = super()._get_obs_extra()
    if self._extract in ["cube"]:
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
                # cube B
                cubeB_pose=vectorize_pose(self.obs_cubeB.pose),
                tcp_to_cubeB_pos=self.obs_cubeB.pose.p - self.tcp.pose.p,
                cubeB_to_place_goal_pos=self.obs_cubeB.pose.p - self.place_goal_site.pose.p,
                cubeB_in_drawer=1 if self.check_cube_in_drawer(self.obs_cubeB) else 0,
            )
    elif self._extract in ["drawer"]:
        pass      
    else:
        obs = super()._get_obs_extra()
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                # place goal
                place_goal_pos=self.place_goal_site.pose.p,
                tcp_to_place_goal_pos=self.place_goal_site.pose.p - self.tcp.pose.p,
                tcp_pose=vectorize_pose(self.tcp.pose),
                # cubeA
                cubeA_pose=vectorize_pose(self.obs_cubeA.pose),
                tcp_to_cubeA_pos=self.obs_cubeA.pose.p - self.tcp.pose.p,
                cubeA_to_place_goal_pos=self.obs_cubeA.pose.p - self.place_goal_site.pose.p,
                # cube B
                cubeB_pose=vectorize_pose(self.obs_cubeB.pose),
                tcp_to_cubeB_pos=self.obs_cubeB.pose.p - self.tcp.pose.p,
                cubeB_to_place_goal_pos=self.obs_cubeB.pose.p - self.place_goal_site.pose.p,
            )
    return obs

def _get_obs_extra_3(self) -> OrderedDict:
    obs = super()._get_obs_extra()
    if self._extract in ["cube"]:
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
                # cube B
                cubeB_pose=vectorize_pose(self.obs_cubeB.pose),
                tcp_to_cubeB_pos=self.obs_cubeB.pose.p - self.tcp.pose.p,
                cubeB_to_place_goal_pos=self.obs_cubeB.pose.p - self.place_goal_site.pose.p,
                cubeB_in_drawer=1 if self.check_cube_in_drawer(self.obs_cubeB) else 0,
                # cube C
                cubeC_pose=vectorize_pose(self.obs_cubeC.pose),
                tcp_to_cubeC_pos=self.obs_cubeC.pose.p - self.tcp.pose.p,
                cubeC_to_place_goal_pos=self.obs_cubeC.pose.p - self.place_goal_site.pose.p,
                cubeC_in_drawer=1 if self.check_cube_in_drawer(self.obs_cubeC) else 0,
            )
    elif self._extract in ["drawer"]:
        pass      
    else:
        obs = super()._get_obs_extra()
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                # place goal
                place_goal_pos=self.place_goal_site.pose.p,
                tcp_to_place_goal_pos=self.place_goal_site.pose.p - self.tcp.pose.p,
                tcp_pose=vectorize_pose(self.tcp.pose),
                # cubeA
                cubeA_pose=vectorize_pose(self.obs_cubeA.pose),
                tcp_to_cubeA_pos=self.obs_cubeA.pose.p - self.tcp.pose.p,
                cubeA_to_place_goal_pos=self.obs_cubeA.pose.p - self.place_goal_site.pose.p,
                # cube B
                cubeB_pose=vectorize_pose(self.obs_cubeB.pose),
                tcp_to_cubeB_pos=self.obs_cubeB.pose.p - self.tcp.pose.p,
                cubeB_to_place_goal_pos=self.obs_cubeB.pose.p - self.place_goal_site.pose.p,
                # cube C
                cubeC_pose=vectorize_pose(self.obs_cubeC.pose),
                tcp_to_cubeC_pos=self.obs_cubeC.pose.p - self.tcp.pose.p,
                cubeC_to_place_goal_pos=self.obs_cubeC.pose.p - self.place_goal_site.pose.p,
            )
    return obs
