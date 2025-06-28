from utils.parse_lib import get_parse
from modules.reward import RewIdentity, RewDSO
# from modules.dso_learn import DSOLearner
from modules.cls_learn import ClsLearner, PredicateCls
from modules.dt_learn import TreeLearner, TreeContDenseLagrangianCls
# from dso.program import Program
from utils.logging import init_logging, log_and_print

import os
import shutil
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
from environment.skill_env import AbsTransit_Push_Multi, AbsTransit_PickPlaceCube, AbsTransit_PickPlaceCubeMulti, AbsTransit_Opendrawer, AbsTransit_Opendrawer_v3, AbsTransit_Pick_Tower, AbsTransit_Pick_Tower_2, AbsTransit_Opendrawer_PickPlaceCubeMulti_2, AbsTransit_Pick_MultiTower, AbsTransit_Pick_Multi_Branch
from environment.maniskill_env.get_env import get_opendrawer, get_opendrawer_v3, opendrawer_dropobs, get_pick_place_cube, get_pick_place_cube_multi, get_drawer_pick_place_cube_multi
from policy.SAC_Lg.sac_lagrangian import SACLag
from policy.SAC_Lg_Gail.sac_lagrangian import SACLagD
# from policy.SAC_Bag_new.sac_bag import SACBag
from policy.SAC_Bag.sac_bag import SACBag
from policy.SAC_Lg_Bag.sac_lagrangian_bag import SACLagBag
from policy.SAC_Lg_Gail_Bag.sac_lagrangian import SACLagBagD
from policy.SAC_Lg_RCE.sac_lagrangian import SACLagRce
from policy.new_rce_sb3.rce import RCE
from policy.new_rce_sb3.replaybuffer import UniformReplayBuffer
from policy.commons.buffer import ReplayBuffferLag, ReplayBufferLagD, ReplayBufferLagBag
from policy.commons.custom_callback_bag import EvalCustomBagCallback
from policy.new_rce_sb3.replaybuffer import ExpertReplayBufferLoad
import policy.awac as AWAC
from synthesize.topdown import topdown, goaldis, check_program

from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from tqdm import tqdm

import matplotlib.pyplot as plt
import copy
import cv2

import pdb

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

USE_GYM_VERSION = 'gymnasium'
ENV_DEBUG = False

# build custom graph for tower 9 block with gripper move away
def build_graph_multi_tower_blockN_new_away(args, skill_graph, obs_transit, obs_seq, block_num=9):
    # add missed attribute
    cur_node = skill_graph.start_node
    while cur_node is not None:
        cur_node.crop_id = None
        cur_node = cur_node.s_node

    # new graph
    new_skill_graph = skill_graph.get_copy()

    # get template node
    tmp_node = new_skill_graph.start_node.s_node.get_copy()

    # init
    start_node = tmp_node.get_copy()
    start_node.reward_fun = None
    start_node.clear_skill()
    start_node.s_node = cur_node
    start_node.node_id = 0
    last_node = start_node

    # 0,1,2,3,4,5,6,7,8
    # 9,10,11,12,13,14,15,16,17
    # 18
    # 19,20,21,22,23,24,25,26,27
    # obs_node_dict = {0:[0,1,2,3,4,5,6,7,8], 
    #                  1:[9,10,11,12,13,14,15,16,17], 
    #                  2:[18,18,18,18,18,18,18,18,18], 
    #                  3:[19,20,21,22,23,24,25,26,27]}
    obs_node_dict = {0:list(np.arange(block_num)),
                     1:list(block_num+np.arange(block_num)),
                     2:[2*block_num]*block_num,
                     3:list(2*block_num+1+np.arange(block_num))}

    print(obs_node_dict)

    node_policy_map = {}
    final_rules = []
    node_num = 1
    for block_id in range(block_num):
        # define
        cur_node = tmp_node.get_copy()
        cur_node.crop_id = block_id
        cur_node.node_id = 1
        cur_node_2 = tmp_node.get_copy()
        cur_node_2.crop_id = block_id
        cur_node_2.node_id = 2
        cur_node_3 = tmp_node.get_copy()
        cur_node_3.crop_id = block_id
        cur_node_3.node_id = 3
        cur_node_4 = tmp_node.get_copy()
        cur_node_4.crop_id = block_id
        cur_node_4.node_id = 4

        # reward function for block 1
        cur_rules = [[f"obs[{obs_node_dict[0][block_id]}]<=0.027050569653511047"]]
        cur_node.reward_fun[0].equ.set_new_rules(cur_rules, None, None)
        cur_node.reward_fun[0].equ.obs_transit = obs_transit

        cur_rules = [[f'obs[{obs_node_dict[1][block_id]}]<=0.03307705000042915']]
        cur_node_2.reward_fun[0].equ.set_new_rules(cur_rules, None, None)
        cur_node_2.reward_fun[0].equ.obs_transit = obs_transit

        cur_rules = [[f'obs[{obs_node_dict[1][block_id]}]<=0.022553522139787674']]
        cur_node_3.reward_fun[0].equ.set_new_rules(cur_rules, None, None)
        cur_node_3.reward_fun[0].equ.obs_transit = obs_transit

        cur_rules = [[f'obs[{obs_node_dict[3][block_id]}]>0.1',
                      f'obs[{obs_node_dict[1][block_id]}]<=0.022553522139787674']]
        cur_node_4.reward_fun[0].equ.set_new_rules(cur_rules, None, None)
        cur_node_4.reward_fun[0].equ.obs_transit = obs_transit

        # connecton
        last_node.s_node = cur_node
        cur_node.s_node = cur_node_2
        cur_node_2.s_node = cur_node_3
        cur_node_3.s_node = cur_node_4
        last_node = cur_node_4

        # general
        node_policy_map[block_id*4] = 0
        node_policy_map[block_id*4+1] = 1
        node_policy_map[block_id*4+2] = 2
        node_policy_map[block_id*4+3] = 3

        final_rules.append(f'obs[{obs_node_dict[1][block_id]}]<=0.022553522139787674')
        node_num += 4

    # start node
    new_skill_graph.start_node = start_node
    new_skill_graph.node_num = node_num

    final_rules = [final_rules]
    final_rew_fun = cur_node_4.reward_fun[0].get_copy()
    final_rew_fun.equ.set_new_rules(cur_rules, None, None)
    final_rew_fun.equ.obs_transit = obs_transit

    # get demo index
    all_demo_idxs = []
    all_rew_fun = []
    cur_node = new_skill_graph.start_node
    while cur_node is not None:
        if cur_node.reward_fun is not None:
            all_rew_fun.append(cur_node.reward_fun[0].get_copy())
        cur_node = cur_node.s_node

    # set demo index
    cur_node = new_skill_graph.start_node
    cur_id = 0
    while cur_node is not None:
        if cur_node.reward_fun is not None:
            cur_node.reward_fun[0].set_train()
            # cur_node.start_idx = all_demo_idxs[cur_id]
            cur_id += 1
        cur_node = cur_node.s_node


    return new_skill_graph, node_policy_map, all_demo_idxs, all_rew_fun, final_rew_fun

def define_env(args, eval=False):
    if 'multitower' in args.env_name:
        task = 'multitower'
        # env = GymToGymnasium(FetchPickAndPlaceConstruction(name=args.env_name, sparse=False, shaped_reward=False, num_blocks=int(args.env_name[-1]), reward_type='sparse', case = 'Multitower', visualize_mocap=False, simple=False))
        env = GymToGymnasium(FetchPickAndPlaceConstruction(name=args.env_name, sparse=False, shaped_reward=False, num_blocks=int(args.env_name[-1]), reward_type='sparse', case = 'Multitower', visualize_mocap=False, simple=True, gripper_away=True))
        # obs_transit = AbsTransit_Pick_Multi()
        obs_transit = AbsTransit_Pick_Tower()
        # obs_transit = AbsTransit_Pick_MultiTower()
    elif 'Pyramid' in args.env_name:
        task = 'pyramid'
        env = GymToGymnasium(FetchPickAndPlaceConstruction(name=args.env_name, sparse=False, shaped_reward=False, num_blocks=int(args.env_name[-1]), reward_type='sparse', case = 'Pyramid', visualize_mocap=False, simple=True, gripper_away=True))
        # obs_transit = AbsTransit_Pick_Multi()
        obs_transit = AbsTransit_Pick_Tower()
    elif 'towerAway' in args.env_name:
        task = 'towerAway'
        env = GymToGymnasium(FetchPickAndPlaceConstruction(name=args.env_name, sparse=False, shaped_reward=False, num_blocks=int(args.env_name[-1]), reward_type='sparse', case = 'Singletower', visualize_mocap=False, stack_only=True, simple=True, gripper_away=True))
        # pdb.set_trace()
        # obs_transit = AbsTransit_Pick_Multi()
        obs_transit = AbsTransit_Pick_Tower()
        # obs_transit = AbsTransit_Pick_Tower_2()

    return task, env, obs_transit

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


def main_debug_eval_iter_multi(args):
    global ENV_DEBUG
    ENV_DEBUG = True

    # tower
    results = {}
    env_names = []
    for i in range(4, 8):
        env_names.append('towerAway'+str(i))
    # pyramid
    for i in range(4, 10):
        env_names.append('Pyramid'+str(i))
    # multi towers
    for i in range(4, 10):
        env_names.append('multitower'+str(i))

    # define policies
    awac_1, awac_2, awac_3, awac_4 = None, None, None, None

    # evaluate for all the environments
    for env_name in env_names:
        # set environment
        args.env_name = env_name
        block_num = int(args.env_name[-1])
        log_and_print('testing {}'.format(args.env_name))

        # define environment
        task, general_env, obs_transit = define_env(args)
        _, general_eval_env, _ = define_env(args, eval=True)
        add_rew_fun = common_skill_reward
        if 'FetchBlockConstruction' in args.env_name:
            add_rew_fun = block_skill_reward
        elif args.disable_rew_add:
            add_rew_fun = common_skill_reward_noadd

        # collect human demonstration
        if args.demo_path is None or not os.path.exists(args.demo_path):
            if not ENV_DEBUG:
                if task == 'custom_block' or task == 'pickmulti':
                    collector = CollectDemos(args.demo_path, traj_len=args.train_traj_len, num_trajectories=10, task=task, img_path=args.img_path, env_name=args.env_name, block_num=int(args.env_name[-1]), debug=False)
                else:
                    collector = CollectDemos(args.demo_path, traj_len=args.train_traj_len, num_trajectories=10, task=task, img_path=args.img_path, env_name=args.env_name, debug=False)
            else:
                collector = CollectDemos(args.demo_path, traj_len=args.train_traj_len, num_trajectories=10, task=task, img_path=args.img_path, env_name=args.env_name, block_num=int(args.env_name[-1]), debug=False)
            obs_seq, obs_imgs = collector.collect(store = args.demo_path is not None)
        else:
            obs_seq = np.load(args.demo_path, allow_pickle=True)
            if args.img_path is None:
                obs_imgs = None
            else:
                with open(args.img_path, 'rb') as f:
                    obs_imgs = pickle.load(f)

        act_seq = [[np.array(each_obs) for each_obs in obs['actions']] for obs in obs_seq]
        obs_seq = [[np.array(each_obs) for each_obs in obs['obs']] for obs in obs_seq]

        # store images
        if not os.path.exists(args.fig_path):
            os.makedirs(args.fig_path)

        # initialize task
        skill_graph = PredicateGraph([RewIdentity()], [0 for _ in obs_seq], [len(each_obs_seq) for each_obs_seq in obs_seq])
        if args.hold_rule:
            cover_rules = []

        # create iterative program
        if args.load_path is not None:
            log_and_print('loading from {}'.format(args.load_path))

            # load graph architecture
            with open(os.path.join(args.load_path, 'graph.pkl'), 'rb') as f:
                skill_graph = pickle.load(f)

            # update graph
            obs_transit.set_num(block_num)
            valid_blocks = list(np.arange(block_num))

            skill_graph, node_policy_map, all_demo_idxs, all_rew_funs, final_reward_fun = build_graph_multi_tower_blockN_new_away(args, skill_graph, obs_transit, obs_seq, block_num=block_num)
            # attempt to build iterative program
            new_reward_fun = skill_graph.get_spec_node(4).reward_fun[0].get_copy()

            tail_node = skill_graph.get_spec_node(4)
            tail_node.s_node = None
            tail_node.stage_node = True

            new_graph = skill_graph.get_copy(skill_drop=True)
            with open('synthesize/program_2.pkl', 'rb') as f:
                condition = pickle.load(f)
                condition.program.val = 0.032
            conditions = {0:condition, 1:None, 2:None, 3:None, 4:None}
            iterative_prog = IterativePredicateGraph(new_graph, conditions, new_reward_fun.equ.obs_transit)
            iterative_prog.valid_blocks = valid_blocks

        # init store
        if not os.path.exists(args.skill_path):
            os.makedirs(args.skill_path)
        if not os.path.exists(args.skill_path+'_fail'):
            os.makedirs(args.skill_path+'_fail')

        # store images
        if not os.path.exists(args.fig_path):
            os.makedirs(args.fig_path)

        # learn policy and subgoal iteratively
        try:
            finetune_policy
        except:
            finetune_policy = {}
        # fail_search = True
        log_and_print('\n-------------------------------------------')
        log_and_print('current task contains:')
        graph_str, cond_str, rew_str = iterative_prog.print_program()
        log_and_print(graph_str)
        log_and_print(rew_str)
        log_and_print('-------------------------------------------\n')

        new_obs_space = gym.spaces.Box(-math.inf, math.inf, shape=[16])
        task_limit = None

        # only for debug
        cur_hold_len = 1

        env = SkillEnvIterNew(general_env, iterative_prog, \
                        traj_len=block_num*100, threshold=args.rew_threshold, hold_len=cur_hold_len, \
                        lagrangian=args.lagrangian_mode, fail_search=False, additional_reward_fun=add_rew_fun, task_limit=task_limit, \
                        new_observation_space=new_obs_space, node_policy_map=node_policy_map, obs_transit=obs_transit)
        env.set_train()

        eval_env = SkillEnvIterNew(general_eval_env, iterative_prog, \
                        traj_len=block_num*100, threshold=args.rew_threshold, hold_len=cur_hold_len, \
                        lagrangian=args.lagrangian_mode, fail_search=False, additional_reward_fun=add_rew_fun, task_limit=task_limit, \
                        new_observation_space=new_obs_space, node_policy_map=node_policy_map, obs_transit=obs_transit)
        eval_env.set_eval()

        # load policies
        if awac_1 is None:
            awac_1, _ = AWAC.define_policy(env, None, 16, 4)
            awac_2, _ = AWAC.define_policy(env, None, 16, 4)
            awac_3, _ = AWAC.define_policy(env, None, 16, 4)
            awac_4, _ = AWAC.define_policy(env, None, 16, 4)
            idx = 8739999
            awac_1.load_state_dict(torch.load(os.path.join(args.load_path, f'policy_0_{idx}.pth')))
            awac_2.load_state_dict(torch.load(os.path.join(args.load_path, f'policy_1_{idx}.pth')))
            awac_3.load_state_dict(torch.load(os.path.join(args.load_path, f'policy_2_{idx}.pth')))
            awac_4.load_state_dict(torch.load(os.path.join(args.load_path, f'policy_3_{idx}.pth')))

        # do the evaluation
        success_num, _ = AWAC.eval_actor_multi(eval_env, [awac_1._actor, awac_2._actor, awac_3._actor, awac_4._actor], 'cuda', 100, 2000, \
                            block_num*100, node_policy_map, None, final_reward_fun, loop_limit=100)

        # store
        results[env_name] = success_num

        np.save(os.path.join(args.skill_path, 'transfer_eval_result.npy'), results, allow_pickle=True)

        log_and_print('for env {}: {}/100'.format(env_name, success_num))

    log_and_print('complete')


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

    import gymnasium as gym
    from environment.data.collect_demos import CollectDemos
    from environment.general_env import GeneralEnv, GeneralDebugEnv, GymToGymnasium
    from environment.skill_env import SkillEnv, SkillEnvIter, SkillEnvPure, SkillEnvIterNew, SkillEnvIterDeepset, CustomMonitor, AbsTransit_Pick, AbsTransit_Push, AbsTransit_Ant, AbsTransit_Block, AbsTransit_Custom_Block, AbsTransit_Pick_Multi, common_skill_reward, block_skill_reward, common_skill_reward_noadd
    from modules.skill import Skill, PredicateGraph, IterativePredicateGraph
    import environment.fetch_block_construction

    main_debug_eval_iter_multi(args)