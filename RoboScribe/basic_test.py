# load gym
import gymnasium as gym
from environment.data.collect_demos import CollectDemos
from environment.general_env import GeneralEnv, GeneralDebugEnv, GymToGymnasium
from environment.skill_env import CustomMonitor, AbsTransit_Pick, AbsTransit_Push, AbsTransit_Ant, AbsTransit_Block, AbsTransit_Custom_Block, AbsTransit_Pick_Multi, AbsTransit_Opendrawer, AbsTransit_Opendrawer_v3, common_skill_reward, block_skill_reward, common_skill_reward_noadd
from environment.skill_env import SkillEnvPure as SkillEnv
from environment.skill_env import SkillEnvIterNew
from modules.skill import Skill, PredicateGraph
import environment.fetch_block_construction

# load learner
from utils.parse_lib import get_parse
from modules.reward import RewIdentity, RewDSO
from modules.cls_learn import ClsLearner, PredicateCls
from modules.dt_learn import TreeLearner, TreeContDenseLagrangianCls
from utils.logging import init_logging, log_and_print

# load environment specific
import os
import cv2
import numpy as np
import pickle
import random
import torch
import math
import commentjson as json
import environment.ant_maze
from environment.Entity_Factored_RL_Env.fetch_push_multi import FetchNPushEnv, FetchNPushObsWrapper
from environment.fetch_custom.get_fetch_env import get_env, get_pickplace_env
from environment.cee_us_env.fpp_construction_env import FetchPickAndPlaceConstruction
from environment.skill_env import AbsTransit_Push_Multi, AbsTransit_PickPlaceCube, AbsTransit_PickPlaceCubeMulti, AbsTransit_Opendrawer_PickPlaceCubeMulti, AbsTransit_Opendrawer_PickPlaceCubeMulti_2, AbsTransit_Pick_Multi_Branch
from environment.skill_env import AbsTransit_MetaWorld, AbsTransit_Pick_Tower
from environment.maniskill_env.get_env import get_opendrawer, get_opendrawer_v3, opendrawer_dropobs, get_pick_place_cube, get_pick_place_cube_multi, get_drawer_pick_place_cube_multi
from policy.SAC_Lg.sac_lagrangian import SACLag
from policy.SAC_Lg_Gail.sac_lagrangian import SACLagD
from policy.SAC_Lg_RCE.sac_lagrangian import SACLagRce
from policy.new_rce_sb3.rce import RCE
from policy.new_rce_sb3.replaybuffer import UniformReplayBuffer
from policy.commons.buffer import ReplayBuffferLag, ReplayBufferLagD
from policy.commons.custom_callback import EvalCustomCallback
from policy.new_rce_sb3.replaybuffer import ExpertReplayBufferLoad
import policy.awac as AWAC

# load RL policies
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from tqdm import tqdm

import matplotlib.pyplot as plt
import copy
import pdb

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

USE_GYM_VERSION = 'gymnasium'
ENV_DEBUG = False
add_obs_fun = None

def define_env(args, eval=False):
    if 'FetchPickAndPlace' in args.env_name:
        task = 'pick'
        # env = gym.make(args.env_name, seed=None, obs_only=True, simple_obs=args.simple_obs)
        env = GeneralEnv(gym.make(args.env_name, max_episode_steps=100, render_mode='rgb_array'))
        obs_transit = AbsTransit_Pick()
    elif 'FetchPush' in args.env_name:
        task = 'push'
        # env = gym.make(args.env_name, seed=None, obs_only=True)
        env = GeneralEnv(gym.make(args.env_name, max_episode_steps=100, render_mode='rgb_array'))
        obs_transit = AbsTransit_Push()
    elif 'AntMaze' in args.env_name:
        task = 'antMaze'
        goal_cand = [28, 29, 30, 31]
        env = GeneralEnv(gym.make(args.env_name, eval=eval, render_mode='rgb_array', goal_cand=goal_cand))
        obs_transit = AbsTransit_Ant()
    elif 'Debug_FetchBlockConstruction' in args.env_name:
        task = 'block'
        # env = GeneralDebugEnv(gym.make(args.env_name[6:], stack_only=False, render_mode='rgb_array'), env_success_rew=1)
        env = GeneralEnv(gym.make(args.env_name[6:], stack_only=True, render_mode='rgb_array'), env_success_rew=1)
        obs_transit = AbsTransit_Block()
        # obs_transit = AbsTransit_Pick()
        # only for debug
        # obs_transit.set_num(2)
        global ENV_DEBUG
        ENV_DEBUG = True
    elif 'FetchBlockConstruction' in args.env_name:
        task = 'block'
        # env = GeneralEnv(gym.make(args.env_name, stack_only=True))
        env = GeneralEnv(gym.make(args.env_name, stack_only=True, render_mode='rgb_array'), env_success_rew=1)
        obs_transit = AbsTransit_Block()
        # only for debug
        obs_transit.set_num(2)
    elif 'BlockCustom' in args.env_name:
        task = 'custom_block'
        env = GeneralEnv(get_env(args.train_traj_len, eval=False, block_num=int(args.env_name[-1])), env_success_rew=0, goal_key='goal')
        obs_transit = AbsTransit_Custom_Block()
        # only for debug
        obs_transit.set_num(int(args.env_name[-1]))
    elif 'BlockPick' in args.env_name:
        task = 'custom_pick'
        env = GeneralEnv(get_pickplace_env(args.train_traj_len, eval=False), env_success_rew=0, goal_key='goal')
        obs_transit = AbsTransit_Custom_Block()
        # only for debug
        obs_transit.set_num(1)
    elif 'pickmultibranch' in args.env_name:
        task = 'pickmultibranch'
        env = GymToGymnasium(FetchPickAndPlaceConstruction(name=args.env_name, sparse=False, shaped_reward=False, num_blocks=2, reward_type='sparse', case = 'PickAndPlaceBranch', visualize_mocap=False, simple=True, gripper_away=True))
        obs_transit = AbsTransit_Pick_Multi_Branch()
    elif 'pickmulti' in args.env_name:
        task = 'pickmulti'
        if args.preset_block_num is not None:
            env = GymToGymnasium(FetchPickAndPlaceConstruction(name=args.env_name, sparse=False, shaped_reward=False, num_blocks=args.preset_block_num, reward_type='sparse', case = 'PickAndPlace', visualize_mocap=False, simple=True))
        else:
            env = GymToGymnasium(FetchPickAndPlaceConstruction(name=args.env_name, sparse=False, shaped_reward=False, num_blocks=int(args.env_name[-1]), reward_type='sparse', case = 'PickAndPlace', visualize_mocap=False, simple=True))
        obs_transit = AbsTransit_Pick_Multi()
    elif 'tower' in args.env_name:
        task = 'tower'
        if args.preset_block_num is not None:
            env = GymToGymnasium(FetchPickAndPlaceConstruction(name=args.env_name, sparse=False, shaped_reward=False, num_blocks=args.preset_block_num, reward_type='sparse', case = 'Singletower', visualize_mocap=False, stack_only=True, simple=True))
        else:
            env = GymToGymnasium(FetchPickAndPlaceConstruction(name=args.env_name, sparse=False, shaped_reward=False, num_blocks=int(args.env_name[-1]), reward_type='sparse', case = 'Singletower', visualize_mocap=False, stack_only=True, simple=True))
        # obs_transit = AbsTransit_Pick_Multi()
        obs_transit = AbsTransit_Pick_Tower()
    elif 'Fetch' in args.env_name and 'Push' in args.env_name:
        task = 'pushmulti'
        env = GymToGymnasium(FetchNPushObsWrapper(FetchNPushEnv(reward_type='sparse', num_objects=int(args.env_name[5]), collisions=True)))
        obs_transit = AbsTransit_Push_Multi()
    elif args.env_name == 'opendrawer_v3':
        task = 'opendrawer'
        env = GymToGymnasium(get_opendrawer_v3())
        obs_transit = AbsTransit_Opendrawer_v3()
    elif args.env_name == 'opendrawer':
        task = 'opendrawer'
        env = GymToGymnasium(get_opendrawer())
        obs_transit = AbsTransit_Opendrawer()
    elif args.env_name == 'pickplacecubemulti_reassign':
        task = 'pickplacecubemulti_reassign'
        env = GymToGymnasium(get_pick_place_cube_multi(solve_cube=2, reassign=True))
        obs_transit = AbsTransit_PickPlaceCubeMulti()
    elif args.env_name == 'pickplacecubemulti':
        task = 'pickplacecubemulti'
        if args.preset_block_num is not None:
            env = GymToGymnasium(get_pick_place_cube_multi(solve_cube=args.preset_block_num))
        elif args.set_block_num is not None:
            env = GymToGymnasium(get_pick_place_cube_multi(solve_cube=args.set_block_num))
        else:
            env = GymToGymnasium(get_pick_place_cube_multi(solve_cube=2))
        obs_transit = AbsTransit_PickPlaceCubeMulti()
    elif args.env_name == 'pickplacecube':
        task = 'pickplacecube'
        env = GymToGymnasium(get_pick_place_cube())
        obs_transit = AbsTransit_PickPlaceCube()
    elif args.env_name == 'drawer_pickplacecubemulti':
        task = 'drawer_pickplacecubemulti'
        env = GymToGymnasium(get_drawer_pick_place_cube_multi(solve_cube=2))
        # obs_transit = AbsTransit_Opendrawer_PickPlaceCubeMulti()
        obs_transit = AbsTransit_Opendrawer_PickPlaceCubeMulti_2()

    if args.stage_reuse:
        assert args.set_block_num is not None, "when setting stage_reuse=True, specific block number of environment need to be set set_block_num="
    if args.set_block_num is not None:
        if args.preset_block_num is not None:
            obs_transit.set_num(args.preset_block_num)
        else:
            obs_transit.set_num(args.set_block_num)

    return task, env, obs_transit

def define_policy(args, env, cur_id=None, param_path=None, device='auto', obs_seq=None, lam_disable=False, env_dims=None):
    # load policy
    if param_path is not None:
        # create policy
        batch_size = args.rl_batch
        if args.policy_type == 'ppo':
            policy = PPO.load(param_path, device=device)
        elif args.policy_type == 'sac':
            policy = SAC.load(param_path, device=device)
        elif args.policy_type == 'sac_lag':
            policy = SACLag.load(param_path, device=device)
        elif args.policy_type == 'sac_lag_d':
            policy = SACLagD.load(param_path, device=device)
        elif args.policy_type == 'sac_lag_rce':
            policy = SACLagRce.load(param_path, device=device)
        elif args.policy_type == 'awac':
            if env is None:
                state_dim = env_dims[0]
                action_dim = env_dims[1]
            else:
                state_dim = env.env.observation_space.shape[0]
                action_dim = env.env.action_space.shape[0]
            policy, _ = AWAC.define_policy(env, None, state_dim, action_dim)
            policy.load_state_dict(torch.load(param_path))

    # From here
    else:
        # init
        env.set_train()
        if 'AntMaze' not in args.env_name:
            # vec_env = make_vec_env(make_env(args, env, True), n_envs=args.n_cpu, vec_env_cls=SubprocVecEnv)
            vec_env = make_vec_env(make_env(args, env, True), n_envs=args.n_cpu)
        else:
            vec_env = make_vec_env(make_env(args, env, True), n_envs=args.n_cpu)

        # create policy
        batch_size = args.rl_batch
        n_steps = args.rl_n_steps
        if args.policy_type == 'ppo':
            policy = PPO("MlpPolicy",
                        vec_env,
                        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                        n_steps=n_steps // args.n_cpu,
                        batch_size=batch_size,
                        n_epochs=10,
                        learning_rate=5e-4,
                        gamma=0.8,
                        verbose=2,
                        tensorboard_log=args.policy_path.format(cur_id),
                        ent_coef=1e-3,
                        device=device)
        elif args.policy_type == 'sac':
            policy = SAC("MlpPolicy", 
                        vec_env, 
                        policy_kwargs=dict(net_arch=[256, 256, 256]), 
                        verbose=2, 
                        batch_size=batch_size, 
                        gamma=0.99, 
                        target_update_interval=2,
                        learning_rate=0.0003, 
                        tau=0.005, 
                        learning_starts=4000, 
                        ent_coef='auto_0.1', 
                        tensorboard_log=args.policy_path.format(cur_id),
                        device=device)
        # hyperparameters for Tower, Pick&Place
        elif args.policy_type == 'sac_lag':
            policy = SACLag(policy="MlpPolicy", 
                        env=vec_env, 
                        policy_kwargs=dict(net_arch=[256, 256, 256]), 
                        verbose=2, 
                        batch_size=batch_size, 
                        gamma=0.95, 
                        target_update_interval=2,
                        learning_rate=0.001, 
                        tau=0.05, 
                        learning_starts=4000,
                        replay_buffer_class=ReplayBuffferLag, 
                        ent_coef='auto_0.1', 
                        tensorboard_log=args.policy_path.format(cur_id),
                        device=device,
                        lam_disable=lam_disable)
        elif args.policy_type == 'sac_lag_d':
            # create expert data
            # exp_data = {'state': np.concatenate(obs_seq, axis=0), 'action': None}
            exp_data = {'state': np.concatenate(obs_seq['obs'], axis=0), 
                        'next_state': np.concatenate(obs_seq['next_obs'], axis=0),
                        'action': np.concatenate(obs_seq['act'], axis=0),
                        'reward': np.concatenate(obs_seq['reward']),
                        'cost': np.concatenate(obs_seq['cost']),
                        'done': np.concatenate(obs_seq['done'])}
            policy = SACLagD(policy="MlpPolicy",
                        exp_data=exp_data,
                        env=vec_env,
                        policy_kwargs=dict(net_arch=[256, 256, 256]),
                        d_kwargs=dict(d_dist=args.d_dist),
                        verbose=2, 
                        batch_size=batch_size, 
                        gamma=0.99, 
                        target_update_interval=2,
                        learning_rate=0.0003, 
                        tau=0.005, 
                        learning_starts=4000,
                        replay_buffer_class=ReplayBufferLagD, 
                        ent_coef='auto_0.1', 
                        tensorboard_log=args.policy_path.format(cur_id),
                        device=device,
                        lam_disable=lam_disable)
        elif args.policy_type == 'sac_rce':
            policy = RCE("MlpPolicy",
                        vec_env,
                        obs_seq,
                        args.n_steps,
                        policy_kwargs=dict(net_arch=[256, 256]),
                        verbose=2, 
                        batch_size=batch_size, 
                        gamma=0.99, 
                        target_update_interval=2,
                        learning_rate=0.0003, 
                        tau=0.005, 
                        learning_starts=4000, 
                        ent_coef=1e-4,
                        gradient_steps=args.n_cpu,
                        tensorboard_log=args.policy_path.format(cur_id),
                        replay_buffer_class=UniformReplayBuffer,
                        replay_buffer_kwargs={'handle_timeout_termination': True})
            
        elif args.policy_type == 'sac_lag_rce':
            # create expert data
            exp_data = {'state': np.concatenate(obs_seq['obs'], axis=0), 
                        'next_state': np.concatenate(obs_seq['next_obs'], axis=0),
                        'action': np.concatenate(obs_seq['act'], axis=0),
                        'reward': np.concatenate(obs_seq['reward']),
                        'cost': np.concatenate(obs_seq['cost']),
                        'done': np.concatenate(obs_seq['done'])}
            policy = SACLagRce(policy="MlpPolicy",
                        exp_data=exp_data,
                        env=vec_env,
                        policy_kwargs=dict(net_arch=[256, 256, 256]), 
                        verbose=2, 
                        batch_size=batch_size, 
                        gamma=0.99, 
                        target_update_interval=2,
                        learning_rate=0.0003, 
                        tau=0.005, 
                        learning_starts=4000,
                        replay_buffer_class=ReplayBufferLagD, 
                        ent_coef='auto_0.1', 
                        tensorboard_log=args.policy_path.format(cur_id),
                        device=device,
                        lam_disable=lam_disable)
            
    return policy

def make_env(args, example_env, train=True):
    def __init():
        global ENV_DEBUG
        _, general_env, _ = define_env(args)
        # copy example env
        skill_graph = example_env.skill_graph
        rew_fun = example_env.rew_fun.get_copy()
        cur_hold_len = example_env.hold_len
        fail_search = example_env.fail_search
        search_node = example_env.search_node
        add_rew_fun = example_env.additional_reward_fun
        add_obs_fun = example_env.add_obs_fun
        if example_env.debug_rew_fun is not None:
            debug_rew_fun = example_env.debug_rew_fun.get_copy()
        else:
            debug_rew_fun = None
        # create environment
        env = SkillEnv(general_env, skill_graph, search_node, rew_fun, traj_len=example_env.traj_len, \
                       threshold=args.rew_threshold, hold_len=cur_hold_len, train=train, \
                       lagrangian=args.lagrangian_mode, fail_search=fail_search, additional_reward_fun=add_rew_fun, env_debug=ENV_DEBUG,\
                       debug_rew_fun=debug_rew_fun, add_obs_fun=add_obs_fun)
        
        return env

    return __init

def learn_policy(args, env, cur_id, resume_policy=None, obs_seq=None, lam_disable=False, eval_env=None):
    if resume_policy:
        policy = resume_policy
        vec_env = make_vec_env(make_env(args, env, True), n_envs=args.n_cpu)
        policy.set_env(vec_env)
        new_logger = configure(args.policy_path.format(cur_id))
        policy.set_logger(new_logger)
    else:
        policy = define_policy(args, env, cur_id=cur_id, obs_seq=obs_seq, lam_disable=lam_disable)

    # train
    if resume_policy:
        assert args.policy_type != 'awac'
        policy.learn(total_timesteps=args.finetune_rl_timesteps, reset_num_timesteps=True)
    else:
        if args.policy_type == 'awac':
            policy, replay_buffer = policy
            AWAC.learn_policy(env, eval_env, policy, replay_buffer, args.eval_traj_len)
        else:
            if eval_env is not None:
                eval_callback = EvalCustomCallback(eval_env, best_model_save_path=args.policy_path.format(cur_id),
                                                log_path=args.policy_path.format(cur_id), eval_freq=5000, n_eval_episodes=100,
                                                deterministic=True, render=False)
                policy.learn(total_timesteps=args.rl_timesteps, callback=eval_callback)
            else:
                policy.learn(total_timesteps=args.rl_timesteps)

    return policy

def eval_policy(args, env, policy, collect_state=False, eval_epochs=100, traj_limit=None):
    global USE_GYM_VERSION
    if traj_limit is None:
        traj_limit = args.eval_traj_len

    # init
    env.set_eval()
    traj_len_list = []
    if collect_state:
        success_state_list = []
        fail_state_list = []

    for epoch in tqdm(range(eval_epochs)):
        # init
        done = False
        obs, _ = env.reset()
        if collect_state:
            state_list = [np.expand_dims(obs, axis=0)]

        # evaluation
        reset_len = env.traj_len - env.traj_id
        for frame_id in range(min(traj_limit, reset_len)):
            action, _ = policy.predict(obs, deterministic=True)
            if USE_GYM_VERSION=='gymnasium':
                obs, reward, done, _, _ = env.step(action)
            else:
                obs, reward, done, _ = env.step(action)
            # if collect_state:
            #     state_list.append(obs)
            if done:
                break

        # store
        if done:
            if collect_state:
                success_state_list.append(state_list[0])
            traj_len_list.append(frame_id+1)
        else:
            if collect_state:
                if args.non_const:
                    fail_state_list.append(np.expand_dims(obs, axis=0))
                else:
                    fail_state_list.append(state_list[0])

    # return
    if collect_state:
        return len(traj_len_list) / float(args.eval_epoch), traj_len_list, success_state_list, fail_state_list
    return len(traj_len_list) / float(args.eval_epoch), traj_len_list


def main_debug(args):
    global ENV_DEBUG
    global add_obs_fun

    log_and_print(f'brief test {args.env_name}')

    # define environment
    task, general_env, obs_transit = define_env(args)
    _, general_eval_env, _ = define_env(args, eval=True)
    add_rew_fun = common_skill_reward
    if 'FetchBlockConstruction' in args.env_name:
        add_rew_fun = block_skill_reward
    elif args.disable_rew_add:
        add_rew_fun = common_skill_reward_noadd

    # additional observation modification
    if 'opendrawer' in args.env_name or args.env_name == 'drawer_pickplacecubemulti':
        add_obs_fun = opendrawer_dropobs
        # add_obs_fun = opendrawer_dropobs_debug
    elif 'metaworld' in args.env_name:
        add_obs_fun = metaworld_dropobs
    
    # collect human demonstration
    assert args.demo_path is not None and os.path.exists(args.demo_path)
    if args.preset_demo_path is not None:
        obs_seq = np.load(args.preset_demo_path, allow_pickle=True)
    else:
        obs_seq = np.load(args.demo_path, allow_pickle=True)
    if args.img_path is None:
        obs_imgs = None
    else:
        with open(args.img_path, 'rb') as f:
            obs_imgs = pickle.load(f)

    act_seq = [[each_obs for each_obs in obs['actions']] for obs in obs_seq]
    obs_seq = [[each_obs for each_obs in obs['obs']] for obs in obs_seq]

    if args.demo_sample_num is not None and args.demo_sample_num < len(obs_seq):
        pick_ids = random.sample(list(np.arange(len(obs_seq))), k=args.demo_sample_num)
        act_seq = [act_seq[i] for i in pick_ids]
        obs_seq = [obs_seq[i] for i in pick_ids]

    # store images
    if not os.path.exists(args.fig_path):
        os.makedirs(args.fig_path)

    # initialize task
    skill_graph = PredicateGraph([RewIdentity()], [0 for _ in obs_seq], [len(each_obs_seq) for each_obs_seq in obs_seq])
    if args.hold_rule:
        cover_rules = []

    # init store
    if not os.path.exists(args.skill_path):
        os.makedirs(args.skill_path)
    if not os.path.exists(args.skill_path+'_fail'):
        os.makedirs(args.skill_path+'_fail')

    # store images
    if not os.path.exists(args.fig_path):
        os.makedirs(args.fig_path)

    # learn policy and subgoal iteratively
    attempt_id = 0
    finish_num = 0
    fail_search = False
    try:
        finetune_policy
    except:
        finetune_policy = {}

    log_and_print('\n-------------------------------------------')
    log_and_print('current attempt: {}'.format(attempt_id))
    log_and_print('current task contains:')
    graph_str, rew_str = skill_graph.print_graph()
    log_and_print(graph_str)
    log_and_print(rew_str)
    log_and_print('-------------------------------------------\n')

    # attemp to learn skill
    parent_node, target_node, task, check_fail = skill_graph.get_search_node()
    if not fail_search and check_fail and not args.disable_fail:
        fail_search = True
    if parent_node.reward_fun is not None:
        # only for debug
        cover_rules = parent_node.reward_fun[-1].equ.rules

    rew_fun_list, st_demo_idxs, gt_demo_idxs = task
    if args.set_traj_limit:
        task_limit = min(np.max(np.array(gt_demo_idxs) - np.array(st_demo_idxs)) * 2, args.train_traj_len)
        log_and_print('curret skill limited to: {}'.format(task_limit))
    else:
        task_limit = None

    policy_list = []
    reward_list = []
    traj_len_list = []

    log_and_print('starting training to current {} function list'.format(len(rew_fun_list)))
    for rew_fun in rew_fun_list:
        exp_data = None

        if isinstance(rew_fun, RewIdentity):
            cur_hold_len = 1
        else:
            cur_hold_len = args.hold_len
        env = SkillEnv(general_env, skill_graph, parent_node, rew_fun, \
                        traj_len=args.train_traj_len, threshold=args.rew_threshold, hold_len=cur_hold_len, \
                        lagrangian=args.lagrangian_mode, fail_search=fail_search, additional_reward_fun=add_rew_fun, task_limit=task_limit, env_debug=ENV_DEBUG,\
                        add_obs_fun=add_obs_fun)
        env.set_train()
        eval_env = SkillEnv(general_eval_env, skill_graph, parent_node, rew_fun.get_copy(), \
                            traj_len=args.train_traj_len, threshold=args.rew_threshold, hold_len=cur_hold_len, \
                            lagrangian=args.lagrangian_mode, fail_search=fail_search, additional_reward_fun=add_rew_fun, task_limit=task_limit, env_debug=ENV_DEBUG, \
                            add_obs_fun=add_obs_fun)
        eval_env.set_eval()

        if parent_node.node_id in finetune_policy:
            log_and_print('finetune policy')
            resume_policy = finetune_policy[parent_node.node_id]
            policy = learn_policy(args, env, attempt_id, resume_policy, \
                                    obs_seq=exp_data, eval_env=eval_env)
        else:
            if args.lagrangian_mode and not isinstance(rew_fun, RewIdentity):
                lam_disable = rew_fun.equ.lag_not_exist()
            else:
                lam_disable = False
            policy = learn_policy(args, env, attempt_id, \
                                    obs_seq=exp_data,
                                    lam_disable=lam_disable, eval_env=eval_env)

        if args.use_best:
            best_model_path = os.path.join(args.policy_path.format(attempt_id), 'best_model.zip')
            if os.path.exists(best_model_path):
                policy = policy.load(best_model_path)

        if ENV_DEBUG:
            pdb.set_trace()

        eval_env = SkillEnv(general_eval_env, skill_graph, parent_node, rew_fun.get_copy(), \
                            traj_len=args.train_traj_len, threshold=args.rew_threshold, hold_len=cur_hold_len, \
                            lagrangian=args.lagrangian_mode, fail_search=fail_search, additional_reward_fun=add_rew_fun, task_limit=task_limit, env_debug=ENV_DEBUG, \
                            add_obs_fun=add_obs_fun)
        reward, traj_len = eval_policy(args, eval_env, policy, collect_state=False, eval_epochs=args.eval_epoch, traj_limit=task_limit)
        policy_list.append(policy)
        reward_list.append(reward)
        traj_len_list.append(traj_len)

        log_and_print('get reward {} for reward function: \n {}'.format(reward, str(env.rew_fun)))

        if reward >= args.success_threshold:
            break

        if parent_node.node_id in finetune_policy:
            del finetune_policy[parent_node.node_id]

        # store graph architecture
        graph_arch = skill_graph.get_copy(skill_drop=True)
        with open(os.path.join(args.skill_path, 'graph.pkl'), 'wb') as f:
            pickle.dump(graph_arch, f)

    log_and_print(f'Test Finish And Success for {args.env_name}')
    log_and_print('\n\n\n')


if __name__ == '__main__':
    # init
    args = get_parse()
    init_logging('logs', args.log_path)

    # init seed (random, numpy, torch)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    main_debug(args)