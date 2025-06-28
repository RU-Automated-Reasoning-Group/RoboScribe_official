from utils.parse_lib import get_parse
from modules.reward import RewIdentity, RewDSO
# from modules.dso_learn import DSOLearner
from modules.cls_learn import ClsLearner, PredicateCls
from modules.dt_learn import TreeLearner, TreeContDenseLagrangianCls
# from dso.program import Program
from utils.logging import init_logging, log_and_print

import os
import cv2
import numpy as np
import pickle
import random
import torch
import math
import commentjson as json
# import environment.ant_maze
# import environment.Entity_Factored_RL_Env
from environment.Entity_Factored_RL_Env.fetch_push_multi import FetchNPushEnv, FetchNPushObsWrapper
from environment.fetch_custom.get_fetch_env import get_env, get_pickplace_env
# from environment.metaworld.get_env import get_metaworld, metaworld_dropobs
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

def opendrawer_dropobs_debug(obs):
    obs_transit = AbsTransit_Opendrawer_PickPlaceCubeMulti_2()
    obs_transit.block_num = 2
    new_obs = obs_transit.do_crop_state(np.expand_dims(obs, 0), 0)[0]
    return new_obs

# only for debug
def debug_build(skill_graph, obs_transit, obs_seq):
    copy_graph = skill_graph.get_copy()
    copy_graph.get_spec_node(3).stage_node = True
    new_skill_graph = PredicateGraph([RewIdentity()], [0 for _ in range(10)], [99 for _ in range(10)])
    new_skill_graph.start_node = copy_graph.start_node
    new_skill_graph.node_num = 4

    # for block 1
    copy_graph = skill_graph.get_copy()
    copy_graph.get_spec_node(3).stage_node = True
    copy_graph.start_node.s_node.node_id = 4
    copy_graph.start_node.s_node.s_node.node_id = 5
    new_skill_graph.get_spec_node(3).s_node = copy_graph.start_node.s_node
    new_skill_graph.node_num = 6

    # replace
    skill_graph = new_skill_graph

    # replace observation id
    update_node = skill_graph.get_spec_node(2)
    update_node.reward_fun[0].equ.obs_transit = obs_transit

    update_node = skill_graph.get_spec_node(3)
    update_rew = update_node.reward_fun[0].equ
    cur_rules = update_rew.rules
    cur_rules = [['obs[2]<=0.03160775209557483']]
    cur_lag = update_rew.lagrangian_rules
    update_node.reward_fun[0].equ.set_new_rules(cur_rules, cur_lag, None)
    update_node.reward_fun[0].equ.obs_transit = obs_transit

    update_node = skill_graph.get_spec_node(4)
    update_rew = update_node.reward_fun[0].equ
    cur_rules = update_rew.rules
    cur_rules = [['obs[1]<=0.02118929074148759', 'obs[2]<=0.03160775209557483']]
    # cur_rules = [['obs[1]<=0.02118929074148759']]
    # new_lag = [['obs[2]<=0.03160775209557483']]
    update_node.reward_fun[0].equ.set_new_rules(cur_rules, None, None)
    update_node.reward_fun[0].equ.obs_transit = obs_transit

    update_node = skill_graph.get_spec_node(5)
    update_rew = update_node.reward_fun[0].equ
    cur_rules = update_rew.rules
    cur_rules = [['obs[3]<=0.03160775209557483', 'obs[2]<=0.03160775209557483']]
    # cur_lag = update_rew.lagrangian_rules
    cur_lag = [['obs[1]<=0.02118929074148759']]
    # cur_lag = None
    update_node.reward_fun[0].equ.lagrangian_rules = None
    update_node.reward_fun[0].equ.set_new_rules(cur_rules, cur_lag, None)
    update_node.reward_fun[0].equ.obs_transit = obs_transit

    # add sequence ids
    # update_node = skill_graph.get_spec_node(5)
    # new_demo_idxs = []
    # update_node.reward_fun[0].set_eval()
    # for each_obs_seq, left_id, obs_id in zip(obs_seq, [0 for _ in obs_seq], [len(each_obs_seq)-2 for each_obs_seq in obs_seq]):
    #     for cur_id in np.arange(obs_id, left_id-2, -1):
    #         if cur_id == left_id-1:
    #             break
    #         if update_node.reward_fun[0].get_reward(np.expand_dims(each_obs_seq[cur_id], 0), 0) < args.rew_threshold:
    #             break
    #     new_demo_idxs.append(cur_id+1)
    # update_node.start_idx = new_demo_idxs
    # last_demo_idxs = (np.array(new_demo_idxs)-1).tolist()
    # pdb.set_trace()

    update_node = skill_graph.get_spec_node(4)
    # new_demo_idxs = []
    # update_node.reward_fun[0].set_eval()
    # for each_obs_seq, left_id, obs_id in zip(obs_seq, [0 for _ in obs_seq], last_demo_idxs):
    #     for cur_id in np.arange(obs_id, left_id-2, -1):
    #         if cur_id == left_id-1:
    #             break
    #         if update_node.reward_fun[0].get_reward(np.expand_dims(each_obs_seq[cur_id], 0), 0) < args.rew_threshold:
    #             break
    #     new_demo_idxs.append(cur_id+1)
    new_demo_idxs = [48, 43, 42, 147, 40, 42, 44, 64, 40, 41, 52, 45, 119, 44, 39, 40, 45, 53, \
                    46, 125, 46, 9, 113, 42, 40, 47, 43, 43, 44, 42, 45, 9, 44, 47, 41, 71, \
                    45, 43, 46, 9, 49, 53, 40, 44, 10, 42, 8, 47, 43, 40]
    update_node.start_idx = new_demo_idxs
    # last_demo_idxs = (np.array(new_demo_idxs)-1).tolist()
    # pdb.set_trace()
    update_node = skill_graph.get_spec_node(5)
    update_node.start_idx = new_demo_idxs

    update_node = skill_graph.get_spec_node(3)
    # new_demo_idxs = []
    # update_node.reward_fun[0].set_eval()
    # for each_obs_seq, left_id, obs_id in zip(obs_seq, [0 for _ in obs_seq], last_demo_idxs):
    #     for cur_id in np.arange(obs_id, left_id-2, -1):
    #         if cur_id == left_id-1:
    #             break
    #         if update_node.reward_fun[0].get_reward(np.expand_dims(each_obs_seq[cur_id], 0), 0) < args.rew_threshold:
    #             break
    #     new_demo_idxs.append(cur_id+1)
    new_demo_idxs = [8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 9, 8, 9, 29, 8, 9, 0, 9, 8, 9, 9, \
                     9, 8, 8, 8, 9, 0, 8, 9, 8, 7, 9, 8, 8, 0, 9, 9, 8, 9, 0, 8, 0, 9, 9, 8]
    update_node.start_idx = new_demo_idxs
    update_node = skill_graph.get_spec_node(2)
    update_node.start_idx = new_demo_idxs
    # pdb.set_trace()

    # change environment setting to drop non-related state dimension (in skill)
    skill_graph.get_spec_node(0).skill.crop_obj_ids = [0]
    skill_graph.get_spec_node(0).skill.reward_fun = skill_graph.get_spec_node(2).reward_fun[-1]
    skill_graph.get_spec_node(2).skill.crop_obj_ids = [0]
    skill_graph.get_spec_node(2).skill.reward_fun = skill_graph.get_spec_node(3).reward_fun[-1]
    skill_graph.get_spec_node(4).skill.crop_obj_ids = [1]
    skill_graph.get_spec_node(4).skill.reward_fun = skill_graph.get_spec_node(5).reward_fun[-1]
    # skill_graph.get_spec_node(5).skill.crop_obj_ids = [1]

    return skill_graph

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
    elif args.env_name == 'metaworld':
        task = 'metaworld'
        env = GymToGymnasium(get_metaworld())
        obs_transit = AbsTransit_MetaWorld()

    if args.stage_reuse:
        assert args.set_block_num is not None, "when setting stage_reuse=True, specific block number of environment need to be set set_block_num="
    if args.set_block_num is not None:
        if args.preset_block_num is not None:
            obs_transit.set_num(args.preset_block_num)
        else:
            obs_transit.set_num(args.set_block_num)

    return task, env, obs_transit

def collect_exp_data_old(obs_seq, obs_imgs, act_seq, st_demo_idxs, gt_demo_idxs):
    exp_obs_seq = []
    exp_next_obs_seq = []
    exp_act_seq = []
    for each_seq, each_img_seq, each_act_seq, s_idx, idx in zip(
        obs_seq, obs_imgs, act_seq, st_demo_idxs, gt_demo_idxs
    ):
        if s_idx >= idx or 1 + s_idx >= min(len(each_seq), idx + 1):
            continue
        end_idx = min(len(each_seq) - 1, idx + 1)
        exp_obs_seq.append(each_seq[s_idx : end_idx + 1])
        exp_next_obs_seq.append(
            each_seq[1 + s_idx : end_idx + 1] + [each_seq[end_idx]]
        )
        # exp_act_seq.append(each_act_seq[s_idx:idx+1])
        exp_act_seq.append(each_act_seq[s_idx : end_idx + 1])
        if len(exp_act_seq[-1]) < len(exp_obs_seq[-1]):
            assert len(exp_obs_seq[-1]) == len(exp_act_seq[-1]) + 1
            exp_act_seq[-1].append(np.zeros_like(each_act_seq[-1]))

    return exp_obs_seq, exp_next_obs_seq, exp_act_seq

def collect_exp_data(obs_seq, obs_imgs, act_seq, st_demo_idxs, gt_demo_idxs, store_imgs=False):
    exp_obs_seq = []
    exp_next_obs_seq = []
    exp_act_seq = []
    exp_img_seq = []
    for each_seq, each_img_seq, each_act_seq, s_idx, idx in zip(
        obs_seq, obs_imgs, act_seq, st_demo_idxs, gt_demo_idxs
    ):
        if s_idx >= idx or 1 + s_idx >= min(len(each_seq), idx + 1):
            continue
        end_idx = min(len(each_seq) - 2, idx)
        exp_obs_seq.append(each_seq[s_idx : end_idx + 1])
        exp_next_obs_seq.append(
            each_seq[1 + s_idx : end_idx + 2]
        )
        if store_imgs:
            exp_img_seq.append(each_img_seq[s_idx:end_idx+1])
        # exp_act_seq.append(each_act_seq[s_idx:idx+1])
        exp_act_seq.append(each_act_seq[s_idx : end_idx + 1])
        if len(exp_act_seq[-1]) < len(exp_obs_seq[-1]):
            assert len(exp_obs_seq[-1]) == len(exp_act_seq[-1]) + 1
            exp_act_seq[-1].append(np.zeros_like(each_act_seq[-1]))

    if store_imgs:
        return exp_obs_seq, exp_next_obs_seq, exp_act_seq, exp_img_seq
    return exp_obs_seq, exp_next_obs_seq, exp_act_seq

def make_exp_data(obs_seq, next_obs_seq, act_seq, rew_fun, add_rew_fun, lagrangian, debug_seq=None):
    # calculate reward, cost and done
    rew_list = []
    cost_list = []
    done_list = []
    if not isinstance(rew_fun, RewIdentity):
        rew_fun.set_train()

    # for each observation
    for each_seq in obs_seq:
        # reward and cost
        rew_seq = []
        cost_seq = []
        for obs in each_seq:
            if isinstance(rew_fun, RewIdentity):
                rew = rew_fun.get_reward(obs, -1)
                cost = 0
                rew = add_rew_fun(rew, cost, rew_fun, lagrangian)
            else:
                rew, cost = rew_fun.get_reward(obs, -1)
                rew = add_rew_fun(rew, cost, rew_fun, lagrangian)
            rew_seq.append(rew)
            cost_seq.append(cost)
        if isinstance(rew_fun, RewIdentity):
            rew_seq[-1] = 1

        rew_list.append(np.array(rew_seq))
        cost_list.append(np.array(cost_seq))
        # done
        try:
            done_seq = np.zeros(len(rew_seq), dtype=np.bool_)
            done_seq[-1] = True
        except:
            pdb.set_trace()
        done_list.append(done_seq)

    return {'obs': obs_seq, 'next_obs': next_obs_seq, 'act': act_seq, 'reward': rew_list, 'cost': cost_list, 'done': done_list}

# get expert data dict
def make_exp_data_dict(obs_seq, act_seq, all_demo_idxs, all_rew_funs, related_policy_ids, crop_ids, add_rew_fun, obs_transit, lagrangian, st_demo_idxs=None, obs_imgs=None, add_on=False):
    # init
    exp_data_dict = {}
    if obs_imgs is not None:
        exp_imgs_dict = {}
    if st_demo_idxs is None:
        new_all_demo_idxs = [np.zeros(len(all_demo_idxs[0]), dtype=np.int32)] + all_demo_idxs
    else:
        new_all_demo_idxs = st_demo_idxs + all_demo_idxs

    for st_demo_idxs, tg_demo_idxs, rew_fun, policy_id, crop_id in \
                zip(new_all_demo_idxs[:-1], new_all_demo_idxs[1:], all_rew_funs, related_policy_ids, crop_ids):
        
        # init
        exp_obs_seq = []
        exp_next_obs_seq = []
        exp_act_seq = []
        exp_stage_dones = []
        rew_list = []
        cost_list = []
        # init reward function
        if not isinstance(rew_fun, RewIdentity):
            rew_fun.set_train()
        if obs_imgs is not None:
            exp_img_seq = []
            cur_id = 0

        # out = False
        for each_obs_seq, each_act_seq, st_idx, tg_idx in zip(obs_seq, act_seq, st_demo_idxs, tg_demo_idxs):
            # stage done
            if tg_idx == -1 or st_idx == -1 or tg_idx <= st_idx:
                if obs_imgs is not None:
                    cur_id += 1
                continue
            if tg_idx+1 >= len(each_obs_seq):
                # pdb.set_trace()
                print('happen, skip for now')
                if obs_imgs is not None:
                    cur_id += 1
                continue
            new_stage_done = np.zeros(tg_idx-st_idx+1, dtype=np.bool_)
            try:
                new_stage_done[-1] = True
            except:
                pdb.set_trace()
                # out = True
                # break
            exp_stage_dones.append(new_stage_done)
            # observation
            if crop_id is not None:
                if add_on:
                    obs_num = tg_idx+1 - st_idx
                    one_hot = np.zeros((obs_num, obs_transit.block_num), dtype=float)
                    one_hot[:, crop_id] = 1.0
                    crop_obs = np.concatenate([np.array(each_obs_seq)[st_idx:tg_idx+1], one_hot], axis=-1)
                else:
                    try:
                        crop_obs = obs_transit.do_crop_state(np.array(each_obs_seq)[st_idx:tg_idx+1], crop_id)
                    except:
                        pdb.set_trace()
            else:
                crop_obs = np.array(each_obs_seq)[st_idx:tg_idx+1]

            exp_obs_seq.append(crop_obs)
            if crop_id is not None:
                if add_on:
                    obs_num = tg_idx+1 - st_idx
                    one_hot = np.zeros((obs_num, obs_transit.block_num), dtype=float)
                    one_hot[:, crop_id] = 1.0
                    crop_next_obs = np.concatenate([np.array(each_obs_seq)[st_idx+1:tg_idx+2], one_hot], axis=-1)
                else:
                    crop_next_obs = obs_transit.do_crop_state(np.array(each_obs_seq)[st_idx+1:tg_idx+2], crop_id)
            else:
                crop_next_obs = np.array(each_obs_seq)[st_idx+1:tg_idx+2]
            exp_next_obs_seq.append(crop_next_obs)

            # only for debug
            # last_crop_obs_pos = crop_next_obs[-1][5:8]
            # last_crop_obs_goal = crop_next_obs[-1][-3:]
            # assert np.sqrt(np.sum((last_crop_obs_pos-last_crop_obs_goal)**2)) <= 0.1923

            try:
                assert crop_next_obs.shape[0] == crop_obs.shape[0]
            except:
                pdb.set_trace()
            # action
            exp_act_seq.append(np.array(each_act_seq[st_idx:tg_idx+1]))
            assert exp_act_seq[-1].shape[0] == crop_next_obs.shape[0]
            # reward
            rew_seq = []
            cost_seq = []
            for obs in each_obs_seq[st_idx+1:tg_idx+2]:
                if isinstance(rew_fun, RewIdentity):
                    rew = rew_fun.get_reward(obs, -1)
                    cost = 0
                    rew = add_rew_fun(rew, cost, rew_fun, lagrangian)
                else:
                    rew, cost = rew_fun.get_reward(obs, -1)
                    rew = add_rew_fun(rew, cost, rew_fun, lagrangian)
                rew_seq.append(rew)
                cost_seq.append(cost)
            if isinstance(rew_fun, RewIdentity):
                rew_seq[-1] = 1
            rew_list.append(np.array(rew_seq))
            cost_list.append(np.array(cost_seq))

            # debug images
            if obs_imgs is not None:
                exp_img_seq.append(obs_imgs[cur_id][st_idx:tg_idx+1])
                cur_id += 1

        # if out:
            # continue
        # if policy_id != 0:
        #     continue

        if policy_id in exp_data_dict:
            exp_data_dict[policy_id]['state'] = np.concatenate([exp_data_dict[policy_id]['state']] + exp_obs_seq, axis=0)
            exp_data_dict[policy_id]['next_state'] = np.concatenate([exp_data_dict[policy_id]['next_state']] + exp_next_obs_seq, axis=0)
            exp_data_dict[policy_id]['action'] = np.concatenate([exp_data_dict[policy_id]['action']] + exp_act_seq, axis=0)
            exp_data_dict[policy_id]['reward'] = np.concatenate([exp_data_dict[policy_id]['reward']] + rew_list)
            exp_data_dict[policy_id]['cost'] = np.concatenate([exp_data_dict[policy_id]['cost']] + cost_list)
            exp_data_dict[policy_id]['done'] = np.concatenate([exp_data_dict[policy_id]['done']] + exp_stage_dones, axis=0)
            if obs_imgs:
                exp_imgs_dict[policy_id] = exp_imgs_dict[policy_id] + exp_img_seq
        else:
            exp_data_dict[policy_id] = {'state': np.concatenate(exp_obs_seq, axis=0), 
                                        'next_state': np.concatenate(exp_next_obs_seq, axis=0),
                                        'action': np.concatenate(exp_act_seq, axis=0),
                                        'reward': np.concatenate(rew_list),
                                        'cost': np.concatenate(cost_list),
                                        'done': np.concatenate(exp_stage_dones, axis=0)}
            if obs_imgs:
                exp_imgs_dict[policy_id] = exp_img_seq

    # create expert data
    if obs_imgs is not None:
        return exp_data_dict, exp_imgs_dict
    return exp_data_dict

def convert_to_CORL_datset(exp_data):
    num_trajs = len(exp_data["obs"])

    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []

    for i in range(num_trajs):
        for j in range(len(exp_data["obs"][i])):
            observations.append(exp_data["obs"][i][j])
            actions.append(exp_data["act"][i][j])
            rewards.append(exp_data["reward"][i][j])
            next_observations.append(exp_data["next_obs"][i][j])
            terminals.append(exp_data["done"][i][j])

    print(f"size of dataset {len(observations)}")

    full_dataset = {}
    full_dataset["observations"] = np.array(observations)
    full_dataset["actions"] = np.array(actions)
    full_dataset["rewards"] = np.array(rewards)
    full_dataset["next_observations"] = np.array(next_observations)
    full_dataset["terminals"] = np.array(terminals)

    return full_dataset

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

def make_rce_expert_buffer(obs_seq, demo_idxs):
    expert_obs = [traj[max(traj_id, 0)] for traj, traj_id in zip(obs_seq, demo_idxs)]
    expert_obs = np.array(expert_obs)
    expert_buffer = ExpertReplayBufferLoad(expert_obs, example_num=expert_obs.shape[0])

    return expert_buffer

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
        # elif args.policy_type == 'sac_lag':
        #     policy = SACLag(policy="MlpPolicy", 
        #                 env=vec_env, 
        #                 policy_kwargs=dict(net_arch=[256, 256, 256]), 
        #                 verbose=2, 
        #                 batch_size=batch_size, 
        #                 gamma=0.99, 
        #                 target_update_interval=2,
        #                 learning_rate=0.0003, 
        #                 tau=0.005, 
        #                 learning_starts=4000,
        #                 replay_buffer_class=ReplayBuffferLag, 
        #                 ent_coef='auto_0.1', 
        #                 tensorboard_log=args.policy_path.format(cur_id),
        #                 device=device,
        #                 lam_disable=lam_disable)
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
            
        elif args.policy_type == 'awac':
            state_dim = env.env.observation_space.shape[0]
            action_dim = env.env.action_space.shape[0]
            if obs_seq is None:
                exp_data = None
            else:
                exp_data = convert_to_CORL_datset(obs_seq)
            policy, replay_buffer = AWAC.define_policy(env, exp_data, state_dim, action_dim)

            return policy, replay_buffer

    return policy

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
                # eval_callback = EvalCallback(eval_env, best_model_save_path=args.policy_path.format(cur_id),
                #                             log_path=args.policy_path.format(cur_id), eval_freq=5000, n_eval_episodes=20,
                #                             deterministic=True, render=False)
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

def collect_state(args, env, policy, skill_graph, search_node, fail_search, store_img=False, collect_epoch=None, task_limit=False):
    global add_obs_fun
    
    print('collecting states...')
    state_list = []
    success_state_list = []
    if store_img:
        img_list = []
    if collect_epoch is None:
        collect_epoch = args.collect_epoch
    if args.enforce_collect:
        enforce_num = collect_epoch
        collect_epoch = 100 * collect_epoch

    for epoch in tqdm(range(collect_epoch)):
        obs, info, success, collect_success, _, state_store, img_store = skill_graph.rollout(env, args.collect_traj_len, search_node.node_id, collect_states=True, \
                                                                                             drop_success=True, fail_search=fail_search, traj_limit_mode=task_limit, add_obs_fun=add_obs_fun)

        if collect_success:
            # store state
            if args.collect_len > 0:
                pdb.set_trace()
                state_list.append(np.concatenate(state_store[-args.collect_len:], 0))
            else:
                state_list.append(np.concatenate(state_store, 0))
            # store images
            img_list.append(img_store)

        # check episode number
        if args.enforce_collect and enforce_num <= len(state_list):
            break

    if args.collect_traj_freq > 1:
        for state_id, each_state_list in enumerate(state_list):
            pick_ids = np.arange(0, each_state_list.shape[0], args.collect_traj_freq)
            state_list[state_id] = each_state_list[pick_ids]
            if store_img:
                img_list[state_id] = [img_list[state_id][pick_id] for pick_id in pick_ids]

    if store_img:
        return state_list, img_list, success_state_list
    return state_list, success_state_list

def store_fail(args, policy_list, rew_fun_list, \
                    general_env, skill_env, skill_graph, \
                    obs_transit, attempt_id, cur_hold_len, \
                    parent_node, search_node, fail_search, task_limit=None):
    global ENV_DEBUG

    # collect fail states
    all_fail_state_list = []
    all_fail_state_imgs = []
    for policy_id, policy in enumerate(policy_list):
        # use the final skill to avoid good states
        if policy is None:
            pass
        elif isinstance(rew_fun_list[policy_id], RewIdentity):
            if ENV_DEBUG:
                final_skill = Skill(policy, rew_fun_list[policy_id], 0, hold_len=1, lagrangian=args.lagrangian_mode, traj_len_limit=task_limit, crop_obj_ids=[0])
            else:
                final_skill = Skill(policy, rew_fun_list[policy_id], 0, hold_len=1, lagrangian=args.lagrangian_mode, traj_len_limit=task_limit)
        else:
            if ENV_DEBUG:
                final_skill = Skill(policy, rew_fun_list[policy_id], args.rew_threshold, hold_len=cur_hold_len, lagrangian=args.lagrangian_mode, traj_len_limit=task_limit, crop_obj_ids=[0])
            else:
                final_skill = Skill(policy, rew_fun_list[policy_id], args.rew_threshold, hold_len=cur_hold_len, lagrangian=args.lagrangian_mode, traj_len_limit=task_limit)

        # attempt to add skill
        if policy is not None:
            parent_node.set_skill(final_skill)
        fail_state_list, fail_state_imgs, success_state_list = collect_state(args, general_env, policy, skill_graph, search_node, fail_search, True, task_limit=task_limit is not None)
        if policy is not None:
            parent_node.clear_skill()

        # augment positive state
        if args.augment_success:
            aug_gt_state_list = []
            for each_obs_seq in success_state_list:
                for cur_id in np.arange(each_obs_seq.shape[0]-1, -2, -1):
                    if cur_id == -1:
                        break
                    if rew_fun_list[policy_id].get_reward(np.expand_dims(each_obs_seq[cur_id], 0), 0) < args.rew_threshold:
                        break
                if cur_id != -1:
                    aug_gt_state_list.append(np.expand_dims(each_obs_seq[cur_id], axis=0))

        all_fail_state_list.append(np.concatenate(fail_state_list, axis=0))
        all_fail_state_imgs += fail_state_imgs
    fail_state_list = np.concatenate(all_fail_state_list, axis=0)
    fail_state_imgs = all_fail_state_imgs

    # store state image
    skill_env.set_train()
    general_id = 0
    cur_fig_path = os.path.join(args.fig_path, str(attempt_id))
    if not os.path.exists(cur_fig_path):
        os.makedirs(cur_fig_path)
    plt.figure()
    for v_id, imgs in enumerate(fail_state_imgs):
        if not os.path.exists(os.path.join(cur_fig_path, str(v_id))):
            os.makedirs(os.path.join(cur_fig_path, str(v_id)))
        for cur_img_id, img in enumerate(imgs):
            obs = fail_state_list[general_id]
            abs_obs = obs_transit.get_abs_obs(np.expand_dims(obs, 0))[0]
            rew_details = []
            for rew_fun in rew_fun_list:
                rew_fun.set_train()
                try:
                    rew_details += rew_fun.equ.execute_details(np.expand_dims(obs, 0))
                except:
                    rew_details += [rew_fun.get_reward(obs, 0)]
            extra_str = str(v_id) + '\n' + \
                        ',  '.join([str(round(element.item(), 3)) for element in abs_obs[:3]]) + '  |  ' + \
                        ',  '.join([str(round(element.item(), 3)) for element in abs_obs[3:]]) + '\n' + \
                        ',  '.join([str(round(r, 3)) for r in rew_details])

            plt.imshow(img)
            plt.title(extra_str)
            plt.savefig(os.path.join(cur_fig_path, str(v_id), '{}.png'.format(cur_img_id)))
            plt.cla()
            general_id += 1
    plt.close()


def generate_new_reward(args, policy_list, rew_fun_list, \
                        general_env, skill_env, skill_graph, \
                        obs_transit, attempt_id, cur_hold_len, \
                        parent_node, search_node, fail_search, \
                        obs_seq, obs_imgs, st_demo_idxs, gt_demo_idxs, cover_rules=[], reverse_label=False, task_limit=None):
    global ENV_DEBUG

    # collect fail states
    all_fail_state_list = []
    all_fail_state_imgs = []
    for policy_id, policy in enumerate(policy_list):
        # use the final skill to avoid good states
        if policy is None:
            pass
        elif isinstance(rew_fun_list[policy_id], RewIdentity):
            if ENV_DEBUG:
                final_skill = Skill(policy, rew_fun_list[policy_id], 0, hold_len=1, lagrangian=args.lagrangian_mode, traj_len_limit=task_limit, crop_obj_ids=[0])
            else:
                final_skill = Skill(policy, rew_fun_list[policy_id], 0, hold_len=1, lagrangian=args.lagrangian_mode, traj_len_limit=task_limit)
        else:
            if ENV_DEBUG:
                final_skill = Skill(policy, rew_fun_list[policy_id], args.rew_threshold, hold_len=cur_hold_len, lagrangian=args.lagrangian_mode, traj_len_limit=task_limit, crop_obj_ids=[0])
            else:
                final_skill = Skill(policy, rew_fun_list[policy_id], args.rew_threshold, hold_len=cur_hold_len, lagrangian=args.lagrangian_mode, traj_len_limit=task_limit)

        # attempt to add skill
        if policy is not None:
            parent_node.set_skill(final_skill)
        fail_state_list, fail_state_imgs, success_state_list = collect_state(args, general_env, policy, skill_graph, search_node, fail_search, True, task_limit=task_limit is not None)
        if policy is not None:
            parent_node.clear_skill()

        # augment positive state
        if args.augment_success:
            aug_gt_state_list = []
            for each_obs_seq in success_state_list:
                for cur_id in np.arange(each_obs_seq.shape[0]-1, -2, -1):
                    if cur_id == -1:
                        break
                    if rew_fun_list[policy_id].get_reward(np.expand_dims(each_obs_seq[cur_id], 0), 0) < args.rew_threshold:
                        break
                if cur_id != -1:
                    aug_gt_state_list.append(np.expand_dims(each_obs_seq[cur_id], axis=0))

        all_fail_state_list.append(np.concatenate(fail_state_list, axis=0))
        all_fail_state_imgs += fail_state_imgs
    fail_state_list = np.concatenate(all_fail_state_list, axis=0)
    fail_state_imgs = all_fail_state_imgs

    # collect ground truth states
    gt_state_list = [np.expand_dims(each_obs_seq[obs_id], 0) for each_obs_seq, st_id, obs_id in zip(obs_seq, st_demo_idxs, gt_demo_idxs) if obs_id > st_id]
    gt_state_list = np.concatenate(gt_state_list)
    if args.augment_success and len(aug_gt_state_list) != 0:
        log_and_print('found augment positive states: {}'.format(len(aug_gt_state_list)))
        aug_gt_state_list = np.concatenate(aug_gt_state_list)
        gt_state_list = np.concatenate([gt_state_list, aug_gt_state_list], axis=0)
    if obs_imgs is None:
        gt_state_imgs = None
    else:
        gt_state_imgs = [each_img_seq[img_id] for each_img_seq, st_id, img_id in zip(obs_imgs, st_demo_idxs, gt_demo_idxs) if img_id > st_id]

    state_data = np.concatenate([fail_state_list, gt_state_list], 0)
    if reverse_label:
        state_label = np.concatenate([np.ones(shape=len(fail_state_list)), np.zeros(shape=len(gt_state_list))], 0)
    else:
        state_label = np.concatenate([np.zeros(shape=len(fail_state_list)), np.ones(shape=len(gt_state_list))], 0)

    # store state image
    skill_env.set_train()
    general_id = 0
    cur_fig_path = os.path.join(args.fig_path, str(attempt_id))
    if not os.path.exists(cur_fig_path):
        os.makedirs(cur_fig_path)
    plt.figure()
    for v_id, imgs in enumerate(fail_state_imgs):
        if not os.path.exists(os.path.join(cur_fig_path, str(v_id))):
            os.makedirs(os.path.join(cur_fig_path, str(v_id)))
        for cur_img_id, img in enumerate(imgs):
            obs = fail_state_list[general_id]
            abs_obs = obs_transit.get_abs_obs(np.expand_dims(obs, 0))[0]
            rew_details = []
            for rew_fun in rew_fun_list:
                rew_fun.set_train()
                try:
                    rew_details += rew_fun.equ.execute_details(np.expand_dims(obs, 0))
                except:
                    rew_details += [rew_fun.get_reward(obs, 0)]
            extra_str = str(v_id) + '\n' + \
                        ',  '.join([str(round(element.item(), 3)) for element in abs_obs[:3]]) + '  |  ' + \
                        ',  '.join([str(round(element.item(), 3)) for element in abs_obs[3:]]) + '\n' + \
                        ',  '.join([str(round(r, 3)) for r in rew_details])

            plt.imshow(img)
            plt.title(extra_str)
            plt.savefig(os.path.join(cur_fig_path, str(v_id), '{}.png'.format(cur_img_id)))
            plt.cla()
            general_id += 1

    if gt_state_imgs is not None:
        if not os.path.exists(os.path.join(cur_fig_path, 'positive')):
            os.makedirs(os.path.join(cur_fig_path, 'positive'))
        pos_id = 0
        for obs, img in zip(gt_state_list, gt_state_imgs):
            abs_obs = obs_transit.get_abs_obs(np.expand_dims(obs, 0))[0]
            rew_details = []
            for rew_fun in rew_fun_list:
                rew_fun.set_train()
                try:
                    rew_details += rew_fun.equ.execute_details(np.expand_dims(obs, 0))
                except:
                    rew_details += [rew_fun.get_reward(obs, 0)]
            extra_str = str(v_id) + '\n' + \
                            ',  '.join([str(round(element.item(), 3)) for element in abs_obs[:3]]) + '  |  ' + \
                            ',  '.join([str(round(element.item(), 3)) for element in abs_obs[3:]]) + '\n' + \
                            ',  '.join([str(round(r, 3)) for r in rew_details])

            plt.imshow(img)
            plt.title(extra_str)
            plt.savefig(os.path.join(cur_fig_path, 'positive', '{}.png'.format(pos_id)))
            plt.cla()
            pos_id += 1
    plt.close()

    # store distribution plot
    if not os.path.exists(os.path.join(cur_fig_path, 'dist')):
        os.makedirs(os.path.join(cur_fig_path, 'dist'))
    fail_abs_obs_list = obs_transit.get_abs_obs(fail_state_list)
    gt_abs_obs_list = obs_transit.get_abs_obs(gt_state_list)

    plt.figure()
    for obs_dim in range(gt_abs_obs_list.shape[1]):
        plt.hist(fail_abs_obs_list[:, obs_dim], color='b', alpha=0.5, label='fail')
        plt.hist(gt_abs_obs_list[:, obs_dim], color='r', alpha=0.5, label='success')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cur_fig_path, 'dist', '{}.png'.format(obs_dim)))
        plt.cla()

    plt.close()

    # learn new reward function
    assert args.predicate_type == 'tree' or args.predicate_type == 'forest'

    with open(os.path.join(args.skill_path, 'gt_state_{}.pkl'.format(attempt_id)), 'wb') as f:
        pickle.dump(gt_state_list, f)
    with open(os.path.join(args.skill_path, 'fail_state_{}.pkl'.format(attempt_id)), 'wb') as f:
        pickle.dump(fail_state_list, f)

    rew_fun, new_demo_idx = get_reward_fun(args, gt_state_list, fail_state_list, obs_transit, obs_seq, st_demo_idxs, gt_demo_idxs, fig_path=cur_fig_path, cover_rules=cover_rules)
    rew_fun = RewDSO(rew_fun)

    # test accuracy
    rew_fun.set_eval()
    pred_results = []
    for state in state_data:
        pred_results.append(float(rew_fun.get_reward(np.expand_dims(state, 0), 0) < args.rew_threshold))
    pred_results = np.array(pred_results)
    accuracy = np.mean(np.abs(state_label - pred_results))
    log_and_print('accuracy of reward functoin is {}'.format(accuracy))

    return rew_fun.equ, new_demo_idx, gt_state_list, fail_state_list

# learn reward function from data
def learn_reward_fun(args, positive_states, negative_states, obs_transit, cover_rules=[], **kwargs):
    # initialize predicate learner
    if args.predicate_type == 'tree' or args.predicate_type == 'forest':
        # assert args.min_samples_mode == 'ratio' and args.min_samples_leaf == 0.1
        if args.min_samples_mode is None or args.min_samples_leaf is None:
            tree_learner = TreeLearner(args.predicate_type, obs_transit, args.dense_type, args.only_old, args.lagrangian_mode, simplify=args.prog_sim, **kwargs)
        elif args.min_samples_mode == 'num':
            tree_learner = TreeLearner(args.predicate_type, obs_transit, args.dense_type, args.only_old, args.lagrangian_mode, simplify=args.prog_sim, \
                                    min_samples_leaf=int(args.min_samples_leaf), **kwargs)
        elif args.min_samples_mode == 'ratio':
            tree_learner = TreeLearner(args.predicate_type, obs_transit, args.dense_type, args.only_old, args.lagrangian_mode, simplify=args.prog_sim, \
                                    min_samples_leaf=args.min_samples_leaf, **kwargs)
        elif args.min_samples_mode == 'pos_ratio':
            tree_learner = TreeLearner(args.predicate_type, obs_transit, args.dense_type, args.only_old, args.lagrangian_mode, simplify=args.prog_sim, \
                                    min_samples_leaf=int(args.min_samples_leaf * len(positive_states)), **kwargs)
        elif args.min_samples_mode == 'min_ratio':
            tree_learner = TreeLearner(args.predicate_type, obs_transit, args.dense_type, args.only_old, args.lagrangian_mode, simplify=args.prog_sim, \
                                    min_samples_leaf=int(args.min_samples_leaf * min(len(positive_states), len(negative_states))), **kwargs)

        # most recent for tower and pick&place
        # tree_learner = TreeLearner(args.predicate_type, obs_transit, args.dense_type, args.only_old, args.lagrangian_mode, simplify=args.prog_sim, \
        #                            min_samples_leaf=int(0.5 * len(positive_states)))
        
        # tree_learner = TreeLearner(args.predicate_type, obs_transit, args.dense_type, args.only_old, args.lagrangian_mode, simplify=args.prog_sim, \
        #                            min_samples_leaf=0.1)
        # tree_learner = TreeLearner(args.predicate_type, obs_transit, args.dense_type, args.only_old, args.lagrangian_mode, simplify=args.prog_sim)
        # tree_learner = TreeLearner(args.predicate_type, obs_transit, args.dense_type, args.only_old, args.lagrangian_mode, simplify=args.prog_sim, \
        #                            min_samples_leaf=args.min_samples_leaf)
        tree_learner.reset()
    else:
        pdb.set_trace()

    # get data
    state_data = np.concatenate([negative_states, positive_states], 0)
    state_label = np.concatenate([np.zeros(shape=len(negative_states)), np.ones(shape=len(positive_states))], 0)

    # learn
    tree_cls = tree_learner.do_learn(state_data, state_label)

    # extract
    rew_fun = tree_cls.extract_dense()
    if args.shift_positive:
        log_and_print('before opt reward function: \n {}'.format(str(rew_fun)))
        # rew_fun.opt_threshold(positive_states, alpha=0.8)
        rew_fun.opt_threshold(positive_states, alpha=args.shift_alpha)

    # update if cover rule
    if args.hold_rule:
        if args.lagrangian_mode:
            rew_fun.comb_rules(cover_rules, comb_method='lagrangian')
        else:
            rew_fun.comb_rules(cover_rules, comb_method='and')
        log_and_print('\n comb with cover rule to update reward function and get: \n{}'.format(str(rew_fun)))

    # prune if require
    if args.prune_rule:
        # rew_fun.prune_rules(positive_states, int(len(positive_states)/2))
        rew_fun.prune_rules(positive_states, int(len(positive_states) * 0.1))
        log_and_print('\n prune rule to drop rare cases and get: \n{}'.format(str(rew_fun)))

    return rew_fun, tree_learner

# get index based on reward function
def get_reward_index(args, reward_fun, demo_obs_seq, start_demo_idxs, end_demo_idxs):
    # get demo indexs for candidate reward functions
    new_demo_idxs = []
    reward_fun.set_eval()
    for each_obs_seq, left_id, obs_id in zip(demo_obs_seq, start_demo_idxs, end_demo_idxs):
        for cur_id in np.arange(obs_id, left_id-2, -1):
            if cur_id == left_id-1:
                break
            if reward_fun.get_reward(np.expand_dims(each_obs_seq[cur_id], 0), 0) < args.rew_threshold:
                break
        new_demo_idxs.append(cur_id+1)

    return new_demo_idxs

# get index based on reward function by forward
def get_reward_index_forward(args, reward_fun, demo_obs_seq, start_demo_idxs, end_demo_idxs):
    # get demo indexs for candidate reward functions
    new_demo_idxs = []
    reward_fun.set_eval()
    obs_transit = reward_fun.equ.obs_transit
    # obs_transit.get_abs_obs(np.expand_dims(each_obs_seq[cur_id], 0))
    # pdb.set_trace()
    demo_id = 0
    for each_obs_seq, left_id, obs_id in zip(demo_obs_seq, start_demo_idxs, end_demo_idxs):
        if left_id == obs_id-1:
            new_demo_idxs.append(left_id)
            continue
        for cur_id in np.arange(left_id, obs_id):
            # if cur_id == left_id-1:
            #     break
            try:
                if reward_fun.get_reward(np.expand_dims(each_obs_seq[cur_id], 0), 0) >= args.rew_threshold:
                    break
            except:
                pdb.set_trace()
        new_demo_idxs.append(cur_id)
        demo_id += 1

    return new_demo_idxs

# pick optimal reward function from learned candidate pool
def get_reward_fun(args, positive_states, negative_states, obs_transit, \
                   demo_obs_seq, start_demo_idxs, end_demo_idxs, opt_pick='min',
                   cover_rules=[], fig_path=None, reward_num=None, comp_thre=0.8):
    reward_num = args.reward_num if reward_num is None else reward_num 

    # learn candidate reward functions
    reward_fun_list = []
    for rew_id in range(reward_num):
        reward_fun, tree_learner = learn_reward_fun(args, positive_states, negative_states, obs_transit, cover_rules)
        # drop out invalid reward (might happen after prune)
        if len(reward_fun.rules) == 0:
            continue

        reward_fun_list.append(RewDSO(reward_fun))
        # log_and_print('{} candidate reward: {}'.format(rew_id, str(reward_fun)))
        if fig_path is not None:
            tree_learner.print_tree(os.path.join(fig_path, 'cls_tree_{}.png'.format(rew_id)))

    # raise error
    if len(reward_fun_list) == 0:
        log_and_print('reward function empty, probability you have set pruning reward which is not valid here')
        pdb.set_trace()

    # get demo indexs for candidate reward functions
    reward_idx_list = []
    for reward_fun in reward_fun_list:
        new_demo_idxs = []
        reward_fun.set_eval()
        for each_obs_seq, left_id, obs_id in zip(demo_obs_seq, start_demo_idxs, end_demo_idxs):
            for cur_id in np.arange(obs_id, left_id-2, -1):
                if cur_id == left_id-1:
                    break
                try:
                    reward_fun.get_reward(np.expand_dims(each_obs_seq[cur_id], 0), 0)
                except:
                    pdb.set_trace()
                if reward_fun.get_reward(np.expand_dims(each_obs_seq[cur_id], 0), 0) < args.rew_threshold:
                    break
            new_demo_idxs.append(cur_id+1)
        reward_idx_list.append(new_demo_idxs)
    reward_idx_list = np.array(reward_idx_list)

    # pick best reward function based on index
    best_demo_idx = reward_idx_list[0]
    best_reward_fun = reward_fun_list[0]
    for cur_idx_list, cur_reward_fun in zip(reward_idx_list[1:], reward_fun_list[1:]):
        if opt_pick == 'min':
            check_idx_list = cur_idx_list < best_demo_idx
        elif opt_pick == 'max':
            check_idx_list = cur_idx_list > best_demo_idx
        else:
            raise NotImplementedError
        if np.mean(check_idx_list.astype(float)) >= comp_thre:
            best_demo_idx = cur_idx_list
            best_reward_fun = cur_reward_fun

    return best_reward_fun.equ, best_demo_idx

# consider stage add
def stage_add(args, skill_graph, eval_env, obs_seq, obs_transit, cur_id, obs_imgs=None, last_states=False, first_states=False):
    global add_obs_fun

    # collect fail states
    collect_epoch = args.collect_epoch
    fail_state_list = []
    fail_imgs_list = []
    for _ in range(collect_epoch * 100):
        _, _, success, _, state_store, img_store, _ = skill_graph.eval_rollout(eval_env, args.eval_traj_len, collect_states=True, env_eval=True, traj_limit_mode=args.set_traj_limit, add_obs_fun=add_obs_fun)
                
        if not success:
            fail_state_list.append([])
            fail_imgs_list.append([])
            # only collect state that succeed the last predicate if required
            if last_states:
                assert not first_states
                state_store = [state_store[-1][-3:]]
                img_store = [img_store[-1][-3:]]
            elif first_states:
                assert not last_states
                # state_store = [[state_store[-1][0]]]
                # img_store = [[img_store[-1][0]]]
                # if len(state_store) == 1:
                #     state_store = [[state_store[-1][0]]]
                #     img_store = [[img_store[-1][0]]]
                # else:
                if len(state_store) == 3:
                    if len(fail_state_list) == 6:
                        pdb.set_trace()
                    state_store = [state_store[-2][-3:]]
                    img_store = [img_store[-2][-3:]]
                else:
                    fail_state_list.pop(-1)
                    fail_imgs_list.pop(-1)
                    continue
            for each_state_store, each_img_store in zip(state_store, img_store):
                fail_state_list[-1] += each_state_store
                fail_imgs_list[-1] += each_img_store
            fail_state_list[-1] = np.concatenate(fail_state_list[-1], 0)

        # if first_states and len(fail_state_list) >= collect_epoch*10:
        if first_states and len(fail_state_list) >= collect_epoch:
            break
        elif not first_states and len(fail_state_list) >= collect_epoch:
            break

    # pdb.set_trace()

    fail_state_list = np.concatenate(fail_state_list, 0)
    # collect success states
    gt_state_list = [np.expand_dims(each_obs_seq[-1], 0) for each_obs_seq in obs_seq]
    gt_state_list = np.concatenate(gt_state_list, 0)
    if obs_imgs is not None:
        gt_imgs_list = [each_obs_img[-1] for each_obs_img in obs_imgs]

    # get stage reward
    end_demo_idxs = [len(traj)-1 for traj in obs_seq]
    cur_fig_path = os.path.join(args.fig_path, str(cur_id), 'stage')
    if not os.path.exists(cur_fig_path):
        os.makedirs(cur_fig_path)

    # pdb.set_trace()

    rew_fun, new_demo_idxs = get_reward_fun(args, gt_state_list, fail_state_list, obs_transit, \
            obs_seq, np.zeros(len(obs_seq), dtype=int).tolist(), end_demo_idxs, opt_pick='min', fig_path = cur_fig_path)
    rew_fun = RewDSO(rew_fun)

    # store distribution plot
    if not os.path.exists(os.path.join(cur_fig_path, 'dist')):
        os.makedirs(os.path.join(cur_fig_path, 'dist'))
    fail_abs_obs_list = obs_transit.get_abs_obs(fail_state_list)
    gt_abs_obs_list = obs_transit.get_abs_obs(gt_state_list)

    # only for debug
    if not os.path.exists(args.skill_path):
        os.makedirs(args.skill_path)
    with open(os.path.join(args.skill_path, 'gt_state_debug.pkl'), 'wb') as f:
        pickle.dump(gt_state_list, f)
    with open(os.path.join(args.skill_path, 'fail_state_debug.pkl'), 'wb') as f:
        pickle.dump(fail_state_list, f)

    # pdb.set_trace()

    # plt.figure()
    # for obs_dim in range(gt_abs_obs_list.shape[1]):
    #     plt.hist(fail_abs_obs_list[:, obs_dim], color='b', alpha=0.5, label='fail')
    #     plt.hist(gt_abs_obs_list[:, obs_dim], color='r', alpha=0.5, label='success')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(cur_fig_path, 'dist', '{}.png'.format(obs_dim)))
    #     plt.cla()

    # plt.close()

    # find order of stage
    current_stages = skill_graph.get_stage_nodes()
    _, next_stage = skill_graph.find_node_idx(new_demo_idxs, True)
    if next_stage is None:
        stage_id = current_stages[-1].node_id
    else:
        for cur_id, stage_node in enumerate(current_stages):
            if stage_node.node_id == next_stage.node_id:
                if cur_id == 0:
                    stage_id = 0
                else:
                    stage_id = current_stages[cur_id-1].node_id
    
    # clear skill
    skill_graph.get_spec_node(stage_id).clear_skill()

    # add new stage
    if args.split_predicate:
        skill_graph.add_stage_node(stage_id, rew_fun.split_rew(gt_state_list), new_demo_idxs)
    else:
        skill_graph.add_stage_node(stage_id, [rew_fun], new_demo_idxs)

    log_and_print('get new stage predicate: \n {}'.format(str(rew_fun)))
    
    # reuse stage
    # if args.stage_reuse:
    #     skill_graph.do_stage_reuse(skill_graph.node_num-1)

    # store gt images
    if obs_imgs is not None:
        if not os.path.exists(os.path.join(cur_fig_path, 'positive')):
            os.makedirs(os.path.join(cur_fig_path, 'positive'))
        for cur_img_id, img in enumerate(gt_imgs_list):
            cv2.imwrite(os.path.join(cur_fig_path, 'positive', '{}.png'.format(cur_img_id)), img)

    # store fail images
    # plt.figure()
    for v_id, imgs in enumerate(fail_imgs_list):
        if not os.path.exists(os.path.join(cur_fig_path, str(v_id))):
            os.makedirs(os.path.join(cur_fig_path, str(v_id)))
        for cur_img_id, img in enumerate(imgs):
            cv2.imwrite(os.path.join(cur_fig_path, str(v_id), '{}.png'.format(cur_img_id)), img)

            # extra_str = str(v_id)
            # plt.imshow(img)
            # plt.title(extra_str)
            # plt.savefig(os.path.join(cur_fig_path, str(v_id), '{}.png'.format(cur_img_id)))
            # plt.cla()
    # plt.close()

# consider creating iterative program
def create_iteractive_program(args, skill_graph, eval_env, obs_seq, obs_transit, cur_id, 
                              diff_env_num_block=None, diff_obs_seq=None):
    global add_obs_fun

    if diff_env_num_block is not None:
        diff_env_skill_graph = skill_graph.get_copy(skill_drop=True)
        diff_env_skill_graph.set_object_centric_diff_env(0, diff_env_num_block)
        diff_env_valid_blocks = np.arange(diff_env_num_block)

    # collect fail states
    collect_epoch = args.collect_epoch
    fail_state_list = []
    fail_imgs_list = []
    for _ in range(collect_epoch * 100):
        _, _, success, _, state_store, img_store, _ = skill_graph.eval_rollout(eval_env, args.eval_traj_len, collect_states=True, env_eval=True, traj_limit_mode=args.set_traj_limit, add_obs_fun=add_obs_fun)
        if not success:
            fail_state_list.append([])
            fail_imgs_list.append([])
            for each_state_store, each_img_store in zip(state_store, img_store):
                fail_state_list[-1] += each_state_store
                fail_imgs_list[-1] += each_img_store
            fail_state_list[-1] = np.concatenate(fail_state_list[-1], 0)
        if len(fail_state_list) >= collect_epoch:
            break
    fail_state_list = np.concatenate(fail_state_list, 0)
    
    # collect success states
    gt_state_list = [np.expand_dims(each_obs_seq[-1], 0) for each_obs_seq in obs_seq]
    gt_state_list = np.concatenate(gt_state_list, 0)

    if args.stage_reuse and obs_transit.block_num is not None:
        start_node = None
        valid_blocks = np.arange(obs_transit.block_num)
        iterative_program = None
        # check whether exists predicate could separate the states
        while True:
            valid, reuse_node_id, new_rew_fun = skill_graph.check_predicate_reuse(gt_state_list, fail_state_list, valid_blocks, \
                                                                     stage_only=True, start_node=start_node)
            
            # fail
            if not valid:
                break
            # attempt to build iterative program
            # new_demo_idxs = get_reward_index(args, new_rew_fun, obs_seq, \
            #                              np.zeros(len(obs_seq), dtype=int).tolist(), \
            #                              np.asarray([len(traj)-1 for traj in obs_seq]))
            new_demo_idxs = None
            # TODO: need to further consider about the graph to iterative program (current directly drop tail rewards)
            if diff_env_num_block is not None and diff_obs_seq is not None:
                diff_new_rew_fun = new_rew_fun.get_copy()
                diff_new_rew_fun.equ.switch_match_diff_env(0, diff_env_num_block, replace=True)
                diff_new_obs_transit = copy.deepcopy(obs_transit)
                diff_new_obs_transit.set_num(diff_env_num_block)

                iterative_program = diff_env_skill_graph.build_iterative_prog(diff_new_rew_fun, new_demo_idxs, diff_env_valid_blocks, diff_obs_seq, diff_new_obs_transit)
            else:
                iterative_program = skill_graph.build_iterative_prog(new_rew_fun, new_demo_idxs, valid_blocks, obs_seq, obs_transit)
            # next
            if iterative_program is not None:
                return iterative_program
            start_node = skill_graph.get_spec_node(reuse_node_id)

    return None

# train iteractive program
def train_iterative_program(args, obs_seq, act_seq, obs_transit, num_blocks, add_rew_fun,
                            iterative_prog, env_kwargs):
    # init
    all_rewards = []
    crop_ids = []
    related_policy_ids = []
    all_demo_idxs = []
    node_policy_map = {}

    # get all reward function
    for b_id in range(num_blocks):
        # init for each object
        iterative_prog.graph.set_object_centric(b_id)
        cur_node = iterative_prog.graph.start_node.s_node
        policy_id = 0
        # store
        while cur_node is not None:
            all_rewards.append(cur_node.reward_fun[0].get_copy())
            crop_ids.append(b_id)
            related_policy_ids.append(policy_id)
            # next
            cur_node = cur_node.s_node
            policy_id += 1

    cur_node = iterative_prog.graph.start_node
    policy_id = 0
    while cur_node is not None and cur_node.s_node is not None:
        node_policy_map[cur_node.node_id] = policy_id
        policy_id += 1
        cur_node = cur_node.s_node

    # get index for demonstration
    start_idxs = [0 for _ in obs_seq]
    end_idxs = [len(each_obs_seq)-1 for each_obs_seq in obs_seq]
    for rew_fun_id, rew_fun in enumerate(all_rewards):
        new_demo_idxs = get_reward_index_forward(args, rew_fun, obs_seq, start_idxs, end_idxs)
        all_demo_idxs.append(new_demo_idxs)
        start_idxs = [max(new_demo_idxs[demo_id], start_idxs[demo_id]) for demo_id in range(len(new_demo_idxs))]
    all_demo_idxs[-1] = [len(traj)-2 for traj in obs_seq]

    # create data
    exp_data = make_exp_data_dict(obs_seq, act_seq, all_demo_idxs, all_rewards, related_policy_ids, crop_ids, add_rew_fun, obs_transit, args.lagrangian_mode)

    datasets = []
    for key in exp_data.keys():
        dataset = {}
        idxs = (np.absolute(exp_data[key]["action"])).max(axis=1) > 0.011
        dataset["observations"] = exp_data[key]["state"][idxs]
        dataset["actions"] = exp_data[key]["action"][idxs]
        dataset["rewards"] = exp_data[key]["reward"][idxs]
        dataset["next_observations"] = exp_data[key]["next_state"][idxs]
        dataset["terminals"] = exp_data[key]["done"][idxs]
        datasets.append(dataset)

    crop_state_dim = datasets[0]['observations'].shape[1]
    action_dim = act_seq[0][0].shape[0]

    # define environment
    eval_env = env_kwargs.pop('eval_env')
    env_kwargs['new_observation_space'] = gym.spaces.Box(-math.inf, math.inf, shape=[crop_state_dim])
    env_kwargs['node_policy_map'] = node_policy_map
    iterative_env = SkillEnvIterNew(**env_kwargs)
    iterative_env.set_train()

    env_kwargs['env'] = eval_env
    env_kwargs['env_rew_ignore'] = False
    eval_iterative_env = SkillEnvIterNew(**env_kwargs)
    eval_iterative_env.set_eval()

    # do the training
    awacs, replay_buffers = [], []
    for dataset in datasets:
        new_awac, new_replay_buffer = AWAC.define_policy(iterative_env, dataset, crop_state_dim, action_dim)
        awacs.append(new_awac)
        replay_buffers.append(new_replay_buffer)

    AWAC.learn_policy_multi(iterative_env, eval_iterative_env, awacs, replay_buffers, args.train_traj_len,
                            node_policy_dict=node_policy_map, update_all=True, store_path=args.skill_path)

    # TODO: need to check success rate here

    return None    

def main_debug(args):
    global ENV_DEBUG
    global add_obs_fun

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
    if args.demo_path is None or not os.path.exists(args.demo_path):
        if not ENV_DEBUG:
            if task == 'custom_block' or task == 'pickmulti' or task == 'tower':
                collector = CollectDemos(args.demo_path, traj_len=args.train_traj_len, num_trajectories=100, task=task, img_path=args.img_path, env_name=args.env_name, block_num=int(args.env_name[-1]), debug=ENV_DEBUG)
            elif task == 'pushmulti':
                collector = CollectDemos(args.demo_path, traj_len=args.train_traj_len, num_trajectories=100, task=task, img_path=args.img_path, env_name=args.env_name, block_num=int(args.env_name[5]), debug=ENV_DEBUG)
            else:
                collector = CollectDemos(args.demo_path, traj_len=args.train_traj_len, num_trajectories=100, task=task, img_path=args.img_path, env_name=args.env_name, debug=ENV_DEBUG)
        else:
            collector = CollectDemos(args.demo_path, traj_len=args.train_traj_len, num_trajectories=100, task=task, img_path=args.img_path, env_name=args.env_name[6:], debug=ENV_DEBUG)
        obs_seq, obs_imgs = collector.collect(store = args.demo_path is not None)
    else:
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

    if args.load_path is not None:
        log_and_print('loading from {}'.format(args.load_path))

        # load graph architecture
        with open(os.path.join(args.load_path, 'graph.pkl'), 'rb') as f:
            skill_graph = pickle.load(f)

        # policy type
        if args.policy_type == 'ppo':
            policy_type = PPO
        elif args.policy_type == 'sac':
            policy_type = SAC
        elif args.policy_type == 'sac_lag':
            policy_type = SACLag
        elif args.policy_type == 'sac_lag_d':
            policy_type = SACLagD
        elif args.policy_type == 'sac_lag_rce':
            policy_type = SACLagRce
        elif args.policy_type == 'sac_rce':
            policy_type = RCE
        elif args.policy_type == 'awac':
            policy_type = AWAC
        else:
            raise NotImplementedError

        # load skills
        policy_dict = {}
        for f_name in os.listdir(args.load_path):
            if 'model' in f_name:
                model_id = int(f_name[:-4].split('_')[-1])
                if args.policy_type == 'awac':
                    env_dims = [general_env.observation_space.shape[0], general_env.action_space.shape[0]]
                    new_policy = define_policy(args, None, env_dims=env_dims, param_path=os.path.join(args.load_path, f_name))
                    # new_policy.load_state_dict(torch.load(os.path.join(args.load_path, f_name, device='cpu')))
                else:
                    new_policy = policy_type.load(os.path.join(args.load_path, f_name), device='cpu')
                policy_dict[model_id] = [new_policy, {'done_thre':args.rew_threshold, 'hold_len':args.hold_len, 'lagrangian':args.lagrangian_mode}]

        # update graph
        skill_graph.update_policys(policy_dict)

        # debug
        # cur_node = skill_graph.get_spec_node(3)
        # cur_node.s_node = skill_graph.get_spec_node(2)
        # cur_node.clear_skill()

        # cur_node.s_node = None
        # cur_node.clear_skill()

        # cur_node = skill_graph.start_node.s_node
        # while cur_node is not None:
        #     cur_node.reward_fun[0].equ.obs_transit = obs_transit
        #     cur_node = cur_node.s_node

        print(skill_graph.print_graph()[0])
        print(skill_graph.print_graph()[1])

        # stage_add(args, skill_graph, general_eval_env, obs_seq, obs_transit, -1, obs_imgs=obs_imgs, last_states=args.last_states, first_states=args.first_states)
        pdb.set_trace()

        # iterative_prog = create_iteractive_program(args, skill_graph, general_eval_env, obs_seq, obs_transit, -1)
        # attempt to add stage if finish (only for debug to comment)
        if skill_graph.check_finish():
            # success_store = []
            # for _ in range(args.final_eval_epoch):
            #     _, _, success, _ = skill_graph.eval_rollout(general_eval_env, args.eval_traj_len, collect_states=False, env_eval=True, traj_limit_mode=args.set_traj_limit, add_obs_fun=add_obs_fun)
            #     success_store.append(float(success))
            # log_and_print('evaluation success rate: {}'.format(np.sum(success_store) / args.final_eval_epoch))

            # # get new predicate and create stage
            # if np.sum(success_store) / args.final_eval_epoch < args.final_goal_threshold:
            if True:
                # attempt to build iterative program
                if args.preset_block_num is not None:
                    new_demo_list = np.load(args.demo_path, allow_pickle=True)
                    new_act_seq = [[np.array(each_obs) for each_obs in obs['actions']] for obs in new_demo_list]
                    new_obs_seq = [[np.array(each_obs) for each_obs in obs['obs']] for obs in new_demo_list]
                    
                    iterative_prog = create_iteractive_program(args, skill_graph, general_eval_env, obs_seq, obs_transit, -1, 
                                                               diff_env_num_block=args.set_block_num, diff_obs_seq=new_obs_seq)
                else:
                    iterative_prog = create_iteractive_program(args, skill_graph, general_eval_env, obs_seq, obs_transit, -1)

                pdb.set_trace()

                if iterative_prog is None:
                    stage_add(args, skill_graph, general_eval_env, obs_seq, obs_transit, -1, obs_imgs=obs_imgs, last_states=args.last_states, first_states=args.first_states)
                else:
                    prog_path = os.path.join(args.skill_path, 'iterative_program')
                    if not os.path.exists(prog_path):
                        os.makedirs(prog_path)
                    iterative_prog.store(prog_path)

                    # attempt to train
                    if args.preset_block_num is not None:
                        obs_seq = new_obs_seq
                        act_seq = new_act_seq
                        # new env
                        new_traj_len = int(args.train_traj_len * (args.set_block_num / args.preset_block_num))
                        args.preset_block_num = None
                        task, general_env, obs_transit = define_env(args)
                        _, general_eval_env, _ = define_env(args, eval=True)

                    pdb.set_trace()

                    env_kwargs = {'env':general_env,
                                  'eval_env': general_eval_env,
                                  'iter_program':iterative_prog,
                                  'traj_len':new_traj_len, 
                                  'threshold':args.rew_threshold, 
                                  'hold_len':args.hold_len,
                                  'lagrangian':args.lagrangian_mode, 
                                  'fail_search':False, 
                                  'additional_reward_fun':add_rew_fun, 
                                  'task_limit':None,
                                  'new_observation_space':None, 
                                  'node_policy_map':None, 
                                  'obs_transit':obs_transit, 
                                  'env_rew_ignore':True}
                    train_iterative_program(args, obs_seq, act_seq, obs_transit, args.set_block_num, add_rew_fun,
                            iterative_prog, env_kwargs)
                    
                    pdb.set_trace()

        # pdb.set_trace()

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
    # fail_search = True
    while not skill_graph.check_finish() and attempt_id < 100:
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
            # cover_rules = parent_node.reward_fun[-1].equ.get_complete_rules()

        rew_fun_list, st_demo_idxs, gt_demo_idxs = task
        if args.set_traj_limit:
            task_limit = min(np.max(np.array(gt_demo_idxs) - np.array(st_demo_idxs)) * 2, args.train_traj_len)
            log_and_print('curret skill limited to: {}'.format(task_limit))
        else:
            task_limit = None

        # get related expert observation
        if args.policy_type in ['sac_lag_d', 'sac_lag_rce', 'sac_rce', 'awac']:
            exp_obs_seq, exp_next_obs_seq, exp_act_seq, exp_img_seq = collect_exp_data(obs_seq, obs_imgs, act_seq, st_demo_idxs, gt_demo_idxs, store_imgs=True)
            # store expert images
            cur_fig_path = os.path.join(args.fig_path, str(attempt_id))
            if not os.path.exists(cur_fig_path):
                os.makedirs(cur_fig_path)
            cur_fig_path = os.path.join(cur_fig_path, 'demo_imgs')
            if not os.path.exists(cur_fig_path):
                os.makedirs(cur_fig_path)
            
            for v_id, each_img_seq in enumerate(exp_img_seq):
                if not os.path.exists(os.path.join(cur_fig_path, str(v_id))):
                    os.makedirs(os.path.join(cur_fig_path, str(v_id)))
                for im_id, img in enumerate(each_img_seq):
                    cv2.imwrite(os.path.join(cur_fig_path, str(v_id), f'{im_id}.png'), img)

        policy_list = []
        reward_list = []
        traj_len_list = []

        log_and_print('starting training to current {} function list'.format(len(rew_fun_list)))
        for rew_fun in rew_fun_list:
            if args.policy_type in ['sac_lag_d', 'sac_lag_rce', 'awac']:
                exp_data = make_exp_data(exp_obs_seq, exp_next_obs_seq, exp_act_seq, rew_fun, add_rew_fun, args.lagrangian_mode)
            elif args.policy_type == 'sac_rce':
                exp_data = make_rce_expert_buffer(obs_seq, [idx+1 for idx in gt_demo_idxs])
            else:
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
                        
            # pdb.set_trace()

            if reward >= args.success_threshold:
                break

        if parent_node.node_id in finetune_policy:
            del finetune_policy[parent_node.node_id]

        # if success store
        success_threshold = args.fail_success_threshold if check_fail else args.success_threshold
        if reward >= success_threshold:
            log_and_print('current attempt success and store skill')
            finish_num += 1
            param_path = os.path.join(args.skill_path, 'model_{}.pth'.format(parent_node.node_id))
            policy.save(param_path)
            new_policy = define_policy(args, env, param_path=param_path, device='cpu')
            if args.traj_limit_mode:
                pdb.set_trace()
                if not fail_search and args.extra_threshold is not None and reward <= args.extra_threshold:
                    pick_rate = max(float(int(success_threshold * 10)) / 10, success_threshold)
                    traj_limit_num = sorted(traj_len_list[-1])[int(pick_rate * args.eval_epoch)-1]
                else:
                    traj_limit_num = np.max(traj_len_list[-1])

                log_and_print('trajectory length limited to be {}'.format(traj_limit_num))
                new_skill = Skill(new_policy, env.rew_fun.get_copy(), args.rew_threshold, hold_len=cur_hold_len, lagrangian=args.lagrangian_mode, traj_len_limit=traj_limit_num)
            elif args.set_traj_limit:
                log_and_print('trajectory length limited to be {}'.format(task_limit))
                new_skill = Skill(new_policy, env.rew_fun.get_copy(), args.rew_threshold, hold_len=cur_hold_len, lagrangian=args.lagrangian_mode, traj_len_limit=task_limit)
            else:
                if ENV_DEBUG:
                    new_skill = Skill(new_policy, env.rew_fun.get_copy(), args.rew_threshold, hold_len=cur_hold_len, lagrangian=args.lagrangian_mode, crop_obj_ids=[0])
                else:
                    new_skill = Skill(new_policy, env.rew_fun.get_copy(), args.rew_threshold, hold_len=cur_hold_len, lagrangian=args.lagrangian_mode)
            parent_node.set_skill(new_skill)
            target_node.reward_fun = [env.rew_fun]
            # update demo index
            if len(rew_fun_list) > 1 and args.split_predicate:
                if target_node.s_node is None:
                    end_demo_idxs = [len(each_obs_seq) for each_obs_seq in obs_seq]
                else:
                    end_demo_idxs = target_node.s_node.start_idx
                new_demo_idxs = get_reward_index(args, env.rew_fun, obs_seq, st_demo_idxs, end_demo_idxs)
                target_node.start_idx = new_demo_idxs
                log_and_print('renew intermediate index to be: \n{}'.format(new_demo_idxs))

            # stage add
            if skill_graph.check_finish():
                success_store = []
                for _ in range(args.final_eval_epoch):
                    _, _, success, _ = skill_graph.eval_rollout(general_eval_env, args.eval_traj_len, collect_states=False, env_eval=True, traj_limit_mode=args.set_traj_limit, add_obs_fun=add_obs_fun)
                    success_store.append(float(success))
                log_and_print('evaluation success rate: {}'.format(np.sum(success_store) / args.final_eval_epoch))
                # get new predicate and create stage
                if np.sum(success_store) / args.final_eval_epoch < args.final_goal_threshold:
                    # attempt to build iterative program
                    if args.preset_block_num is not None:
                        new_demo_list = np.load(args.demo_path, allow_pickle=True)
                        new_act_seq = [[each_obs for each_obs in obs['actions']] for obs in new_demo_list]
                        new_obs_seq = [[each_obs for each_obs in obs['obs']] for obs in new_demo_list]
                        
                        iterative_prog = create_iteractive_program(args, skill_graph, general_eval_env, obs_seq, obs_transit, -1, 
                                                                diff_env_num_block=args.set_block_num, diff_obs_seq=new_obs_seq)
                    else:
                        iterative_prog = create_iteractive_program(args, skill_graph, general_eval_env, obs_seq, obs_transit, -1)

                    if iterative_prog is None:
                        stage_add(args, skill_graph, general_eval_env, obs_seq, obs_transit, -1, obs_imgs=obs_imgs, last_states=args.last_states, first_states=args.first_states)
                    else:
                        prog_path = os.path.join(args.skill_path, 'iterative_program')
                        if not os.path.exists(prog_path):
                            os.makedirs(prog_path)
                        iterative_prog.store(prog_path)

                        # attempt to train
                        if args.preset_block_num is not None:
                            obs_seq = new_obs_seq
                            act_seq = new_act_seq
                            # new env
                            new_traj_len = int(args.train_traj_len * (args.set_block_num / args.preset_block_num))
                            args.preset_block_num = None
                            task, general_env, obs_transit = define_env(args)
                            _, general_eval_env, _ = define_env(args, eval=True)

                        env_kwargs = {'env':general_env,
                                    'eval_env': general_eval_env,
                                    'iter_program':iterative_prog,
                                    'traj_len': new_traj_len,
                                    'threshold':args.rew_threshold, 
                                    'hold_len':args.hold_len,
                                    'lagrangian':args.lagrangian_mode, 
                                    'fail_search':False, 
                                    'additional_reward_fun':add_rew_fun, 
                                    'task_limit':None,
                                    'new_observation_space':None, 
                                    'node_policy_map':None, 
                                    'obs_transit':obs_transit, 
                                    'env_rew_ignore':True}
                        train_iterative_program(args, obs_seq, act_seq, obs_transit, args.set_block_num, add_rew_fun,
                                iterative_prog, env_kwargs)
                        
                        log_and_print('iterative program training finished ...')

                        break

        # if fail define new task
        else:
            # store fail
            param_path = os.path.join(args.skill_path+'_fail', 'model_fail_{}.pth'.format(attempt_id))
            policy.save(param_path)

            # check whether
            # if np.sum(np.array(gt_demo_idxs) - np.array(st_demo_idxs)) <= 0:
            if not np.any(np.array(gt_demo_idxs) > np.array(st_demo_idxs)):
                log_and_print('task fail and no complete skill is found')
                # do store
                store_fail(args, policy_list, rew_fun_list, \
                            general_env, env, skill_graph, \
                            obs_transit, attempt_id, cur_hold_len, \
                            parent_node, target_node, fail_search, task_limit=task_limit)
                break

            # temporarily add skill to collect fail states
            rew_fun, new_demo_idxs, gt_state_list, fail_state_list = \
                      generate_new_reward(args, policy_list, rew_fun_list, \
                                          general_env, env, skill_graph, \
                                          obs_transit, attempt_id, cur_hold_len, \
                                          parent_node, target_node, fail_search, \
                                          obs_seq, obs_imgs, np.zeros_like(np.array(gt_demo_idxs)), gt_demo_idxs, cover_rules=[], task_limit=task_limit)
            rew_fun = RewDSO(rew_fun)

            log_and_print('current attempt fail')
            log_and_print('get new reward function: \n{}'.format(str(rew_fun)))
            log_and_print('get new intermediate index: \n{}'.format(new_demo_idxs))

            # find correct order of node
            last_node, next_node = skill_graph.find_node_idx(new_demo_idxs)
            if last_node.node_id != parent_node.node_id:
                log_and_print('switch skill with node {}'.format(last_node.node_id))
            
            # combine cover rules
            if args.hold_rule and last_node.reward_fun is not None:
                if last_node.skill is None:
                    assert last_node.node_id == parent_node.node_id
                cover_rules = last_node.reward_fun[-1].equ.get_complete_rules()
                if args.lagrangian_mode:
                    rew_fun.comb_rules(cover_rules, comb_method='lagrangian')
                else:
                    rew_fun.comb_rules(cover_rules, comb_method='and')
                log_and_print('\n comb with cover rule to update reward function and get: \n{}'.format(str(rew_fun)))

            last_node.clear_skill()
            # define new task
            if args.split_predicate:
                skill_graph.add_node(rew_fun.split_rew(gt_state_list), new_demo_idxs, last_node, next_node, add_stage=attempt_id==0)
            else:
                skill_graph.add_node([rew_fun], new_demo_idxs, last_node, next_node, add_stage=attempt_id==0)

            # drop environment reward if fail
            # if 'FetchPickAndPlace' not in args.env_name and attempt_id == 0:
            if attempt_id == 0:
                skill_graph.drop_env_rew()

        # next
        attempt_id += 1

        # store graph architecture
        graph_arch = skill_graph.get_copy(skill_drop=True)
        with open(os.path.join(args.skill_path, 'graph.pkl'), 'wb') as f:
            pickle.dump(graph_arch, f)

    log_and_print('finish and complete for {} skills'.format(finish_num))


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

    if True or 'FetchPickAndPlace' in args.env_name or 'FetchPush' in args.env_name:
        import gymnasium as gym
        from environment.data.collect_demos import CollectDemos
        from environment.general_env import GeneralEnv, GeneralDebugEnv, GymToGymnasium
        from environment.skill_env import CustomMonitor, AbsTransit_Pick, AbsTransit_Push, AbsTransit_Ant, AbsTransit_Block, AbsTransit_Custom_Block, AbsTransit_Pick_Multi, AbsTransit_Opendrawer, AbsTransit_Opendrawer_v3, common_skill_reward, block_skill_reward, common_skill_reward_noadd
        from environment.skill_env import SkillEnvPure as SkillEnv
        from environment.skill_env import SkillEnvIterNew
        from modules.skill import Skill, PredicateGraph
        import environment.fetch_block_construction
    else:
        import gym
        from environment.data.collect_demos_gym import CollectDemos
        from environment.general_env_gym import GeneralEnv
        from environment.skill_env_gym import SkillEnv, CustomMonitor, AbsTransit_Pick, AbsTransit_Push, AbsTransit_Ant, AbsTransit_Block
        from modules.skill_gym import Skill, PredicateGraph
        # special env
        import environment.fetch_block_construction_gym
        USE_GYM_VERSION = 'gym'

    main_debug(args)