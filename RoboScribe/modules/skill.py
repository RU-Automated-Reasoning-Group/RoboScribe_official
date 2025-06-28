import os
import pdb
import numpy as np
import copy
import math
import pickle
from modules.reward import RewIdentity
from synthesize.dsl import Box
from synthesize.topdown import topdown
import matplotlib.pyplot as plt

class Skill:
    def __init__(self, policy, reward_fun, done_thre, hold_len=1, lagrangian=False, traj_len_limit=math.inf, 
                 crop_obj_ids=None, obs_transit=None, policy_id=None):
        # policy
        self.policy = policy
        # reward
        self.reward_fun = reward_fun
        self.reward_fun.set_eval()
        self.traj_len_limit = traj_len_limit
        # others
        self.done_thre = done_thre
        self.hold_len = hold_len
        assert self.hold_len >= 1
        self.lagrangian = lagrangian
        # only for debug for now
        self.crop_obj_ids = crop_obj_ids
        self.obs_transit = obs_transit
        self.policy_id = policy_id

    def reset(self):
        pass

    def get_copy(self):
        return Skill(copy.deepcopy(self.policy), self.reward_fun.get_copy(), self.done_thre, self.hold_len)

    def rollout(self, env, obs, skill_limit, collect_states=False, drop_success=False, add_obs_fun=None, collect_states_fun=None, collect_actions=False):
        # debug
        if not isinstance(self.reward_fun, RewIdentity):
            assert not self.reward_fun.train

        # init
        rollout_num = 0
        hold_step = 0
        if collect_states:
            state_list = []
            img_list = []
            if collect_actions:
                action_list = []

        for frame_id in range(skill_limit):
            # do clip
            if self.crop_obj_ids is not None:
                # act_obs = env.env.get_custom_obs(self.crop_obj_ids[0])
                act_obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), self.crop_obj_ids[0])[0]
                # act_obs = self.obs_transit.add_idx_state(np.expand_dims(obs, 0), self.crop_obj_ids[0])[0]
            else:
                act_obs = obs

            if add_obs_fun is not None:
                act_obs = add_obs_fun(act_obs)

            if self.policy_id is not None:
                action, _ = self.policy.predict(act_obs, deterministic=True, policy_id=self.policy_id)
            else:
                action, _ = self.policy.predict(act_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # drop if environment success
            if drop_success and reward == env.env_success_rew:
                terminated = truncated = False
                if collect_states:
                    if collect_states_fun is not None:
                        state_list.append(collect_states_fun())
                    else:
                        state_list.append(obs)
                    img_list.append(env.render())
                    if collect_actions:
                        action_list.append(action)
                break

            reward = self.reward_fun.get_reward(obs, reward)
            if reward >= self.done_thre:
                hold_step += 1
            else:
                hold_step = 0

            if collect_states:
                if collect_states_fun is not None:
                    state_list.append(collect_states_fun())
                else:
                    state_list.append(obs)
                img_list.append(env.render())
                if collect_actions:
                    action_list.append(action)
            rollout_num += 1
            if hold_step >= self.hold_len:
                break
        
        # terminate if success
        if hold_step >= self.hold_len:
            terminated = truncated = True
        
        if collect_states:
            if collect_actions:
                return obs, reward, terminated, truncated, info, rollout_num, state_list, img_list, action_list
            return obs, reward, terminated, truncated, info, rollout_num, state_list, img_list
        return obs, reward, terminated, truncated, info, rollout_num
    
# predicate node (state node and out edge policy)
class PredicateNode:
    def __init__(self, reward_fun, start_idx, node_id, skill=None, s_node=None, f_node=None, traj_len_limit=None, state_cls=None, stage_node=False, crop_id=None):
        # node structure
        self.s_node = s_node
        self.f_node = f_node
        self.state_cls = state_cls
        self.stage_node = stage_node

        # others
        self.reward_fun = reward_fun
        self.start_idx = start_idx
        self.node_id = node_id

        # skill
        self.skill = skill
        self.traj_len_limit = traj_len_limit

        # for combined loop
        self.crop_id = crop_id

    def pick_branch(self, obs):
        assert self.state_cls is not None
        succeed = self.state_cls.get_reward(obs, None)
        return succeed

    def get_task_list(self):
        return [self.reward_fun, self.start_idx]

    def set_policy(self, policy):
        policy_model, kwargs = policy
        if isinstance(self.s_node.reward_fun, RewIdentity):
            kwargs['hold_len'] = 1
        kwargs['traj_len_limit'] = self.traj_len_limit
        self.skill = Skill(policy_model, self.s_node.reward_fun[-1].get_copy(), **kwargs)

    def set_skill(self, skill):
        self.skill = skill
        self.traj_len_limit = skill.traj_len_limit

    def clear_skill(self):
        self.skill = None
        self.traj_len_limit = None

    # only copy self
    def get_copy(self, skill_drop=False):
        # copy reward function
        if self.reward_fun is not None:
            new_reward_fun = [reward_fun.get_copy() for reward_fun in self.reward_fun]
        else:
            new_reward_fun = None
        if self.state_cls is not None:
            new_state_cls = self.state_cls.get_copy()
        else:
            new_state_cls = None
        # copy skill
        if self.skill is None or skill_drop:
            new_skill = None
        else:
            new_skill = copy.deepcopy(self.skill)
        # new node
        new_node = PredicateNode(new_reward_fun, copy.deepcopy(self.start_idx), \
                                 self.node_id, skill=new_skill, traj_len_limit=self.traj_len_limit, \
                                 state_cls=new_state_cls, stage_node=self.stage_node, crop_id=self.crop_id)

        return new_node

    # check solvable if change obj id
    def check_reuse(self, positive_states, negative_states, valid_blocks, do_adj=False, threshold=-0.01):
        # check object
        cur_stage_objs = self.reward_fun[0].equ.get_obj_id()
        if len(cur_stage_objs) > 1:
            return False, None
        
        # check change object id
        new_reward = self.reward_fun[0].get_copy()
        for block_id in valid_blocks:
            new_reward.equ.switch_match(block_id, replace=True)
            # only for debug
            new_reward.set_train()
            pos_res = []
            neg_res = []
            # pos_val = []
            # neg_val = []
            for p_s in positive_states:
                res = new_reward.get_reward(np.expand_dims(p_s, 0), 0) 
                pos_res.append(float(res[0])>threshold)
                # pos_val.append(float(res[0]))
            for n_s in negative_states:
                res = new_reward.get_reward(np.expand_dims(n_s, 0), 0) 
                neg_res.append(float(res[0])>threshold)
                # neg_val.append(float(res[0]))

            # pdb.set_trace()
            # plt.figure()
            # plt.hist(pos_val, color='blue', alpha=0.5, weights=np.ones(len(pos_val)) / len(pos_val))
            # plt.hist(neg_val, color='red', alpha=0.5, weights=np.ones(len(neg_val)) / len(neg_val))
            # plt.savefig('debug_imgs/iterative/iterative_{}.png'.format(block_id))

            if np.sum(pos_res) == len(pos_res) and np.sum(neg_res) == 0:
                return True, new_reward

            # attempt to adjust threshold
            if do_adj:
                adj_new_reward = new_reward.get_copy()
                adj_new_reward.adjust_equ(positive_states)
                pos_res = []
                neg_res = []
                for p_s in positive_states:
                    res = adj_new_reward.get_reward(np.expand_dims(p_s, 0), 0) 
                    pos_res.append(float(res))
                for n_s in negative_states:
                    res = adj_new_reward.get_reward(np.expand_dims(n_s, 0), 0) 
                    neg_res.append(float(res))

                if np.sum(pos_res) == len(pos_res) and np.sum(neg_res) == 0:
                    return True, adj_new_reward

        return False, None

    # find the correct index for specific object trajectories
    def get_traj_index(self, obj_seq, start_idxs, end_idxs, rew_id=0):
        reward_fun = self.reward_fun[rew_id].get_copy()
        new_demo_idxs = []
        reward_fun.set_eval()
        for each_obs_seq, left_id, obs_id in zip(obj_seq, start_idxs, end_idxs):
            for cur_id in np.arange(obs_id, left_id-2, -1):
                if cur_id == left_id-1:
                    break
                if not reward_fun.get_reward(np.expand_dims(each_obs_seq[cur_id], 0), 0):
                    break
            new_demo_idxs.append(cur_id+1)

        return new_demo_idxs

    # set crop id
    def set_crop_id(self, obj_id):
        if self.skill is not None:
            self.skill.crop_obj_ids = [obj_id]
        if self.reward_fun is not None:
            for rew_fun in self.reward_fun:
                rew_fun.equ.switch_match(obj_id, replace=True)
                self.crop_id = obj_id

    # set crop id for a different environment
    def set_crop_id_diff_env(self, obj_id, goal_num_block):
        if self.skill is not None:
            self.skill.crop_obj_ids = [obj_id]
        if self.reward_fun is not None:
            for rew_fun in self.reward_fun:
                rew_fun.equ.switch_match_diff_env(obj_id, goal_num_block, replace=True)
                self.crop_id = obj_id

# graph for predicate
class PredicateGraph:
    def __init__(self, start_rew_fun, start_idx, end_idx, start_node=None, disable_fail=False):
        if start_node is not None:
            self.start_node = start_node
        else:
            self.start_node = PredicateNode(None, start_idx, node_id=0)
            next_node = PredicateNode(start_rew_fun, end_idx, node_id=1)
            self.start_node.s_node = next_node
        self.node_num = 2
        self.disable_fail = disable_fail

    def set_object_centric(self, object_id):
        cur_node = self.start_node
        while cur_node is not None:
            cur_node.set_crop_id(object_id)
            cur_node = cur_node.s_node

    # set object centric for a different environment (same type with different number of blocks)
    def set_object_centric_diff_env(self, object_id, goal_num_block):
        cur_node = self.start_node
        while cur_node is not None:
            cur_node.set_crop_id_diff_env(object_id, goal_num_block)
            cur_node = cur_node.s_node

    def get_node_num(self):
        node_num = 0
        cur_node = self.start_node
        while cur_node is not None:
            node_num += 1
            cur_node = cur_node.s_node
        return node_num

    # helper to get search node
    def _get_search_node(self, cur_node):
        # back if None
        if cur_node is None:
            return None, None
        
        # find target node if no policy is trained
        if cur_node.skill is None and cur_node.s_node is not None:
            return cur_node, None

        # success node first
        parent_node, fail_check = self._get_search_node(cur_node.s_node)
        if parent_node is not None:
            fail_check = False if fail_check is None else fail_check
            return parent_node, fail_check
        
        # fail node next:
        if not self.disable_fail:
            parent_node, fail_check =  self._get_search_node(cur_node.f_node)
            if parent_node is not None:
                # fail_check = True if fail_check is None else fail_check
                fail_check = True
                return parent_node, fail_check
            
        # otherwise
        return None, None

    # get the next node for skill train
    def get_search_node(self, start_id=None):
        start_node = self.start_node if start_id is None else self.get_spec_node(start_id)
        parent_node, fail_search = self._get_search_node(start_node)
        if parent_node is not None:
            reward_fun, end_idx = parent_node.s_node.get_task_list()
            end_idx = [idx-1 for idx in end_idx]
            _, start_idx = parent_node.get_task_list()
            return parent_node, parent_node.s_node, [reward_fun, start_idx, end_idx], fail_search
        else:
            return None, None, None, None

    # get specific node based on id
    def get_spec_node(self, node_id):
        # init
        visited_ids = {}
        cur_node = self.start_node
        while cur_node is not None:
            visited_ids[cur_node.node_id] = False
            cur_node = cur_node.s_node
        visited_ids[self.start_node.node_id] = True
        queue = [self.start_node]

        # traverse
        while len(queue) != 0:
            cur_node = queue.pop(0)
            if cur_node.node_id == node_id:
                return cur_node
            if cur_node.s_node is not None and not visited_ids[cur_node.s_node.node_id]:
                queue.append(cur_node.s_node)
                visited_ids[cur_node.s_node.node_id] = True
            if cur_node.f_node is not None and not visited_ids[cur_node.f_node.node_id]:
                queue.append(cur_node.f_node)
                visited_ids[cur_node.f_node.node_id] = True
        
        return None

    # get all stage nodes
    def get_stage_nodes(self):
        stage_nodes = []
        cur_node = self.start_node

        while cur_node is not None:
            if cur_node.stage_node:
                stage_nodes.append(cur_node)
            cur_node = cur_node.s_node

        return stage_nodes

    # get the last node
    def get_last_node(self):
        cur_node = self.start_node
        while cur_node is not None:
            if cur_node.s_node is None:
                return cur_node
            cur_node = cur_node.s_node

        return None

    # stage reuse
    def do_stage_reuse(self, stage_id):
        # get stage nodes
        stage_nodes = self.get_stage_nodes()
        cur_stage_node = self.get_spec_node(stage_id)
        cur_stage_objs = cur_stage_node.reward_fun[0].equ.get_obj_id()
        # only for debug
        if len(cur_stage_objs) > 1:
            return False

        # find match stage nodes
        found_node = None
        for each_id, each_node in enumerate(stage_nodes):
            if each_node.node_id == stage_id:
                continue
            # check whether match
            if cur_stage_node.reward_fun[0].equ.check_match(each_node.reward_fun[0].equ.rules):
                found_node = [each_id, each_node]
                break

        # If found match, reuse stage
        if found_node is None:
            return False
        
        # get found node
        found_id, found_stage_node = found_node
        if found_id == 0:
            found_start_node = self.start_node.s_node
        else:
            found_start_node = stage_nodes[found_id-1].s_node

        # get current node
        cur_start_node = self.start_node
        while cur_start_node is not None:
            if cur_start_node.s_node is not None and cur_start_node.s_node.node_id == cur_stage_node.node_id:
                break
            cur_start_node = cur_start_node.s_node
        
        # copy
        while found_start_node.node_id != found_stage_node.node_id:
            # copy node
            next_new_node = found_start_node.get_copy(skill_drop=True)
            next_new_node.node_id = self.node_num
            next_new_node.reward_fun[0].equ.switch_match(cur_stage_objs[0], replace=True)
            next_new_node.s_node = cur_start_node.s_node
            cur_start_node.s_node = next_new_node
            # next
            found_start_node = found_start_node.s_node
            cur_start_node = next_new_node
            self.node_num += 1

        return True

    # stage reuse based on positive and negative state rather than reward
    def check_predicate_reuse(self, positive_states, negative_states, valid_blocks, stage_only=True, start_node=None):
        # get stage nodes
        if start_node is None:
            cur_node = self.start_node.s_node
        else:
            cur_node = start_node

        # check all predicates
        reusable = False
        while cur_node is not None:
            if not stage_only or cur_node.stage_node:
                reusable, new_rew_fun = cur_node.check_reuse(positive_states, negative_states, valid_blocks)
                if reusable:
                    break
            cur_node = cur_node.s_node

        if not reusable:
            return False, None, None
        else:
            return True, cur_node.node_id, new_rew_fun

    # synthesize programmatic condition based on demo
    def synthesize_cond(self, reward_fun, valid_blocks, obs_seq, threshold=-0.01, branch=False):
        if branch:
            all_inputs = self.get_all_inputs_branch(reward_fun, valid_blocks, obs_seq, threshold)
        else:
            all_inputs = self.get_all_inputs(reward_fun, valid_blocks, obs_seq, threshold)

        # consider values
        all_vals = reward_fun.equ.get_reward_values()

        # synthesize condition
        for val in all_vals:
            conditions = {}
            for block_id in valid_blocks:
                conditions[block_id] = topdown(all_inputs[block_id], val)

            for k in conditions:
                if conditions[k] is not None:
                    return conditions

        return conditions

    # get all inputs
    def get_all_inputs(self, reward_fun, valid_blocks, obs_seq, threshold=-0.01):
        # init
        test_reward_fun = reward_fun.get_copy()
        test_reward_fun.set_train()
        obs_transit = reward_fun.equ.obs_transit
        all_inputs = {block_id:[] for block_id in valid_blocks}

        # split inputs
        for obs_seq_id, each_obs_seq in enumerate(obs_seq):
            # get block label for each state
            tmp_block_labels = np.zeros((len(valid_blocks), len(each_obs_seq)))
            block_labels = {block_id: np.zeros(len(each_obs_seq)) for block_id in valid_blocks}
            block_orders = [None for _ in valid_blocks]

            # get true label
            for block_idx, block_id in enumerate(valid_blocks):
                test_reward_fun.equ.switch_match(block_id, replace=True)
                for obs_id, obs in enumerate(each_obs_seq):
                    tmp_block_labels[block_idx][obs_id] = test_reward_fun.get_reward(obs, 0)[0] > threshold

            # get handle label
            for block_idx, block_id in enumerate(valid_blocks):
                label_sum = np.sum(tmp_block_labels, axis=0)
                block_order = int(np.max(label_sum * (1 - tmp_block_labels[block_idx])))
                block_labels[block_id] = label_sum == np.max(block_order)
                block_orders[block_order] = block_id

            # TODO: found invalid demo, skip for now
            skip_data = False
            for v in block_orders:
                if v is None:
                    print(obs_seq_id)
                    skip_data = True
                    break
            if skip_data:
                continue

            # try:
            #     # assert block_orders[0] == 0 and block_orders[1] == 1
            #     assert block_orders[0] == 1 and block_orders[1] == 2
            # except:
            #     print('should handle later')
            #     continue

            # get all inputs
            for obs_idx, obs in enumerate(each_obs_seq):
                boxes = {}
                all_boxes = []
                for block_id in valid_blocks:
                    # block_obs = np.squeeze(obs_transit.do_crop_state(np.expand_dims(obs, 0), block_id, return_dict=True), axis=0)
                    b = Box(str(block_id))
                    block_obs = obs_transit.do_crop_state(np.expand_dims(obs, 0), block_id, return_dict=True)
                    block_pos = block_obs['block'][0]
                    block_goal = block_obs['block_goal'][0]
                    b.set_attribute(block_pos[0], block_pos[1], block_pos[2], block_goal[0], block_goal[1], block_goal[2])
                    boxes[block_id] = b
                    all_boxes.append(b)
                
                # get label
                for block_order_idx in range(len(block_orders)):
                    gt_block_id = block_orders[block_order_idx]
                    # positive
                    has_true = False
                    for block_id in block_orders[:block_order_idx+1]:
                        if block_labels[block_id][obs_idx]:
                            all_inputs[gt_block_id].append({"target": boxes[block_id], "all_box": all_boxes, "result": True, 'idx': '{}_{}'.format(obs_seq_id, obs_idx)})
                            has_true = True
                            if not block_labels[gt_block_id][obs_idx]:
                                all_inputs[gt_block_id].append({"target": boxes[gt_block_id], "all_box": all_boxes, "result": False, 'idx': '{}_{}'.format(obs_seq_id, obs_idx)})
                            break
                    if not has_true:
                        if block_labels[gt_block_id][obs_idx]:
                            pdb.set_trace()
                        all_inputs[gt_block_id].append({"target": boxes[gt_block_id], "all_box": all_boxes, "result": False, 'idx': '{}_{}'.format(obs_seq_id, obs_idx)})
        
        return all_inputs

    # get all inputs for if branch
    def get_all_inputs_branch(self, reward_fun_list, block_list, valid_blocks, obs_seq, threshold=-0.01):
        # init
        assert len(reward_fun_list) == len(block_list)
        test_reward_fun_list = []
        for reward_fun in reward_fun_list:
            test_reward_fun = reward_fun.get_copy()
            test_reward_fun.set_train()
            test_reward_fun_list.append(test_reward_fun)
        obs_transit = test_reward_fun.equ.obs_transit
        all_inputs = {block_id:[] for block_id in valid_blocks}

        # split inputs
        for obs_seq_id, each_obs_seq in enumerate(obs_seq):
            # get block label for each state
            tmp_block_labels = np.zeros((len(test_reward_fun_list), len(each_obs_seq)))
            block_labels = {block_id: np.zeros(len(each_obs_seq)) for block_id in block_list}
            block_orders = [None for _ in test_reward_fun_list]

            # get true label
            for block_id, test_reward_fun in zip(block_list, test_reward_fun_list):
                for obs_id, obs in enumerate(each_obs_seq):
                    tmp_block_labels[block_id][obs_id] = test_reward_fun.get_reward(obs, 0)[0] > threshold

            # get handle label
            for block_id, test_reward_fun in zip(block_list, test_reward_fun_list):
                label_sum = np.sum(tmp_block_labels, axis=0)
                block_order = int(np.max(label_sum * (1 - tmp_block_labels[block_id])))
                block_labels[block_id] = label_sum == np.max(block_order)
                block_orders[block_order] = block_id

            pdb.set_trace()

            # get all inputs
            for obs_idx, obs in enumerate(each_obs_seq):
                boxes = {}
                all_boxes = []
                for block_id in block_list:
                    b = Box(str(block_id))
                    block_obs = obs_transit.do_crop_state(np.expand_dims(obs, 0), block_id, return_dict=True)
                    block_pos = block_obs['block'][0]
                    block_goal = block_obs['block_goal'][0]
                    b.set_attribute(block_pos[0], block_pos[1], block_pos[2], block_goal[0], block_goal[1], block_goal[2])
                    boxes[block_id] = b
                    all_boxes.append(b)
                
                # TODO
                pdb.set_trace()

                # get label
                for block_order_idx in range(len(block_orders)):
                    gt_block_id = block_orders[block_order_idx]
                    # positive
                    has_true = False
                    for block_id in block_orders[:block_order_idx+1]:
                        if block_labels[block_id][obs_idx]:
                            all_inputs[gt_block_id].append({"target": boxes[block_id], "all_box": all_boxes, "result": True, 'idx': '{}_{}'.format(obs_seq_id, obs_idx)})
                            has_true = True
                            if not block_labels[gt_block_id][obs_idx]:
                                all_inputs[gt_block_id].append({"target": boxes[gt_block_id], "all_box": all_boxes, "result": False, 'idx': '{}_{}'.format(obs_seq_id, obs_idx)})
                            break
                    if not has_true:
                        if block_labels[gt_block_id][obs_idx]:
                            pdb.set_trace()
                        all_inputs[gt_block_id].append({"target": boxes[gt_block_id], "all_box": all_boxes, "result": False, 'idx': '{}_{}'.format(obs_seq_id, obs_idx)})
        
        return all_inputs


    # attempt to create iterative program
    def build_iterative_prog(self, new_rew_fun, new_demo_idxs, valid_blocks, obs_seq, obs_transit):
        # consider the simplest situation (all stages have the same formate)
        stage_nodes = self.get_stage_nodes()
        found_stage_node = None
        for stage in stage_nodes:
            # if not stage.reward_fun[0].equ.check_match(new_rew_fun.equ.rules):
            #     return None
            if stage.reward_fun[0].equ.check_match(new_rew_fun.equ.rules):
                found_stage_node = stage
                break
        if found_stage_node is None:
            return None

        # pre_iter_graph
        found_obj_ids = found_stage_node.reward_fun[0].equ.get_obs_id()
        pre_iter_node = None
        pre_iter_start_node = None

        # pick the first stage and create general graph
        new_start_node = self._get_copy(self.start_node, {}, skill_drop=True)
        cur_node = new_start_node
        new_node_num = 0
        while cur_node is not None:
            # if cur_node.node_id == stage_nodes[0].node_id:
            #     cur_node.s_node = None
            if cur_node.node_id == found_stage_node.node_id:
                # define pre_iter_graph
                cur_node.s_node = None
                if pre_iter_node is not None:
                    pre_iter_start_node = new_start_node
                    new_start_node = self._get_copy(pre_iter_node, {}, skill_drop=True)
                    pre_iter_node.s_node = None
                
            elif cur_node.stage_node:
                # check whether object match
                cur_obj_ids = cur_node.reward_fun[0].equ.get_obs_id()
                if obs_transit.obj_sep(found_obj_ids, cur_obj_ids):
                    pre_iter_node = cur_node
            cur_node = cur_node.s_node
            new_node_num += 1
        new_graph = PredicateGraph(None, None, None, new_start_node)
        new_graph.disable_fail = self.disable_fail
        new_graph.node_num = new_node_num
        if pre_iter_start_node is not None:
            new_pre_iter_graph = PredicateGraph(None, None, None, pre_iter_start_node)
            new_pre_iter_graph.disable_fail = self.disable_fail
            new_pre_iter_graph.node_num = new_node_num

        # synthesize the condition of object selection for each graph
        conditions = self.synthesize_cond(new_rew_fun, valid_blocks, obs_seq)

        if pre_iter_start_node is not None:
            return IterativePredicateGraph(new_graph, conditions, obs_transit=new_rew_fun.equ.obs_transit, valid_blocks=valid_blocks,
                                           pre_iter_graph=new_pre_iter_graph, reuse_pre_iter_graph=True)
        else:
            return IterativePredicateGraph(new_graph, conditions, obs_transit=new_rew_fun.equ.obs_transit, valid_blocks=valid_blocks)


    # check whether searching over fail branch
    def check_fail_search(self):
        _, fail_search = self._get_search_node(self.start_node)
        return fail_search

    # find head of fail branch
    def find_head_fail(self, branch_id):
        cur_node = self.start_node
        ignore_id = None
        found_node = None
        while cur_node is not None:
            if cur_node.node_id == branch_id:
                return found_node
            if cur_node.node_id == ignore_id:
                found_node = None

            if cur_node.f_node is not None:
                if cur_node.s_node is not None:
                    ignore_id = cur_node.s_node.node_id
                found_node = cur_node
                cur_node = cur_node.f_node
            else:
                cur_node = cur_node.s_node

        return None

    # find most suitable positive of current index (only consider main branch)
    def find_node_idx(self, search_idx, stage_only=False, comp_thre=0.8):
        cur_node = self.start_node
        while cur_node.s_node is not None:
            # directly to next
            if stage_only and not cur_node.s_node.stage_node:
                cur_node = cur_node.s_node
                continue

            # find idx
            next_idx = cur_node.s_node.start_idx
            # idx_comp = np.array(search_idx) < np.array(next_idx)
            idx_comp = np.array(search_idx) <= np.array(next_idx)
            if np.mean(idx_comp.astype(float)) >= comp_thre:
                return cur_node, cur_node.s_node
            
            # next
            cur_node = cur_node.s_node

        return cur_node, None


    # add node containing new reward function
    def add_node(self, new_rew_fun, new_start_id, parent_node, target_node, add_stage=False):
        new_node = PredicateNode(new_rew_fun, new_start_id, node_id=self.node_num, stage_node=add_stage)
        self.node_num += 1
        
        parent_node.s_node = new_node
        new_node.s_node = target_node

        return new_node

    # add fail node
    def add_fail_node(self, fail_rew_fun, fail_start_id, cur_node):
        fail_node = PredicateNode(fail_rew_fun, fail_start_id, node_id=self.node_num)
        self.node_num += 1

        cur_node.f_node = fail_node
        fail_node.s_node = cur_node.s_node

        return fail_node

    # add stage node
    def add_stage_node(self, prev_node_id, new_rew_fun, new_start_id):
        new_node = PredicateNode(new_rew_fun, new_start_id, node_id=self.node_num, stage_node=True)
        self.node_num += 1

        # add at the front
        prev_node = self.get_spec_node(prev_node_id)
        new_node.s_node = prev_node.s_node
        prev_node.s_node = new_node

    # drop environment reward (only search main branch)
    def drop_env_rew(self):
        cur_node = self.start_node
        while cur_node is not None:
            # pick
            next_node = cur_node.s_node
            # test
            if next_node != None and isinstance(next_node.reward_fun[-1], RewIdentity):
                cur_node.s_node = None
                break
            # next
            cur_node = cur_node.s_node

    # check whether finish
    def check_finish(self):
        parent_node, _ = self._get_search_node(self.start_node)
        if parent_node is None:
            return True
        return False

    # print graph: currently only work for one fail branch
    def print_graph(self):
        cur_node = self.start_node
        first_branch_str = ''
        second_branch_str = ''
        reward_str = ''

        while cur_node is not None:
            # get fail first
            new_fail_str = ''
            new_success_str = ''
            if cur_node.reward_fun is None:
                reward_str += 'S{}: NA / {} / num:{} \n'.format(cur_node.node_id, cur_node.skill is not None, np.mean(cur_node.start_idx))
            else:
                reward_str += 'S{}: {} / {} / num:{} \n'.format(cur_node.node_id, str(cur_node.reward_fun[0]), cur_node.skill is not None, np.mean(cur_node.start_idx))
            if cur_node.f_node is not None:
                next_node = cur_node.s_node
                assert next_node is not None

                cur_f_node = cur_node.f_node
                while cur_f_node.node_id != next_node.node_id:
                    if cur_f_node.reward_fun is None:
                        reward_str += 'S{}: NA / {} / num:{} \n'.format(cur_f_node.node_id, cur_f_node.skill is not None, np.mean(cur_f_node.start_idx))
                    else:
                        reward_str += 'S{}: {} / {} / num:{} \n'.format(cur_f_node.node_id, str(cur_f_node.reward_fun[0]), cur_f_node.skill is not None, np.mean(cur_f_node.start_idx))
                    if cur_f_node.skill is None:
                        new_fail_str += 'S{}'.format(cur_f_node.node_id) + '--->'
                    else:
                        new_fail_str += 'S{}'.format(cur_f_node.node_id) + '===>'
                    cur_f_node = cur_f_node.s_node
                new_fail_str = new_fail_str[:-4] + '   '

            # graph string
            new_str = 'S{}'.format(cur_node.node_id)
            if cur_node.skill is None:
                arrow_char = '-'
            else:
                arrow_char = '='
            if len(new_fail_str) > 0:
                new_success_str = new_str + arrow_char * (len(new_fail_str) - len(new_str) - 1) + '>'
            else:
                new_success_str = new_str + arrow_char * 2 + '>'
            if len(new_fail_str) < len(new_success_str):
                new_fail_str += ' ' * (len(new_success_str) - len(new_fail_str))
            first_branch_str += new_success_str
            second_branch_str += new_fail_str

            # next
            cur_node = cur_node.s_node
        
        return first_branch_str + '\n' + second_branch_str, reward_str

    # rollout (only for one fail branch)
    def rollout(self, env, total_len, stop_id, collect_states=False, drop_success=False, fail_search=False, traj_limit_mode=False, start_id=None, exist_obs=None, add_obs_fun=None, **kwargs):
        # reset
        if exist_obs is None:
            obs, info = env.reset(**kwargs)
            info['drop'] = False
        else:
            obs = exist_obs
            info = {'drop':False}
        traj_id = 0
        state_store = []
        img_store = []
        start_node = self.start_node if start_id is None else self.get_spec_node(start_id)

        # rollout skill graph
        # rollout success
        success = True
        # whether success on current task
        task_success = None
        # whether success for collection
        collect_success = None
        head_node = self.find_head_fail(stop_id)
        cur_node = start_node
        while cur_node is not None and cur_node.node_id != stop_id:
            cur_skill = cur_node.skill
            if cur_skill is None:
                pdb.set_trace()
                assert isinstance(cur_node.reward_fun[-1], RewIdentity)
                break
            cur_skill.reset()

            # check fail branch
            if fail_search and cur_node.f_node is not None:
                pdb.set_trace()
                pick_s_branch = cur_node.pick_branch(obs)
                if not pick_s_branch:
                    cur_node = cur_node.f_node
                    if cur_node.node_id == stop_id:
                        collect_success = False
                        success = True
                        continue
                    elif cur_node.skill is None:
                        collect_success = False
                        success = False
                        break
                    continue
                else:
                    if cur_node.node_id == head_node.node_id:
                        success = False
                        collect_success = False
                        break

            # trajectory limit
            if traj_limit_mode:
                traj_len_limit = min(total_len-traj_id, cur_skill.traj_len_limit)
            else:
                # pdb.set_trace()
                traj_len_limit = total_len - traj_id
            if collect_states:
                obs, reward, terminated, truncated, info, traj_len, new_states, new_imgs = cur_skill.rollout(env, obs, traj_len_limit, collect_states=True, drop_success=drop_success, add_obs_fun=add_obs_fun)
                state_store += [np.expand_dims(state, 0) for state in new_states]
                img_store += new_imgs
            else:
                obs, reward, terminated, truncated, info, traj_len = cur_skill.rollout(env, obs, traj_len_limit, drop_success=drop_success, add_obs_fun=add_obs_fun)
            traj_id += traj_len

            # drop success
            info['drop'] = False
            if drop_success and traj_len < traj_len_limit and not terminated:
                info['drop'] = True
                success = False
                collect_success = False
                break

            # check task success
            task_success = total_len > traj_id and terminated
            # check rollout success
            if traj_limit_mode:
                success = total_len > traj_id
            else:
                # pdb.set_trace()
                success = total_len > traj_id and task_success

            # success branch
            if success:
                cur_node = cur_node.s_node
                if cur_node is not None and cur_node.node_id == stop_id:
                    if task_success:
                        collect_success = False
                    else:
                        collect_success = True
            # fail
            else:
                # Whether collect previous fail
                if cur_node.s_node is not None and cur_node.s_node.node_id == stop_id and not task_success:
                    collect_success = True
                else:
                    collect_success = False
                break

        # only for debug
        if success:
            assert cur_node.node_id == stop_id

        if self.start_node.node_id != stop_id:
            assert collect_success is not None
        if collect_states:
            return obs, info, success, collect_success, traj_id, state_store, img_store
        return obs, info, success, traj_id

    # rollout for evaluation
    def eval_rollout(self, env, total_len, collect_states=False, pick_eval=False, env_eval=False, traj_limit_mode=False, exist_obs=None, add_obs_fun=None, collect_states_fun=None, collect_actions=False, **kwargs):
        # reset
        if exist_obs is None:
            obs, info = env.reset(**kwargs)
        else:
            obs = exist_obs
            info = {}
        # init
        traj_id = 0
        if collect_states_fun is None:
            state_store = [[np.expand_dims(obs, 0)]]
        else:
            state_store = [[np.expand_dims(collect_states_fun(), 0)]]
        action_store = []

        img_store = [[env.render()]]
        rew_store = []
        if pick_eval:
            pick_store = []
            success_store = []
            pick_imgs = []
        if env_eval:
            env_success = False

        # rollout skill graph
        success = True
        task_success = None
        cur_node = self.start_node
        while cur_node.skill is not None:
            cur_skill = cur_node.skill
            if cur_skill is None:
                pdb.set_trace()
                assert isinstance(cur_node.reward_fun[-1], RewIdentity)
                break
            cur_skill.reset()

            # check fail branch
            if cur_node.f_node is not None and not self.disable_fail:
                pick_s_branch = cur_node.pick_branch(obs)
                if pick_eval:
                    pick_store.append(pick_s_branch)
                    pick_imgs.append(env.render())
                elif not pick_s_branch:
                    cur_node = cur_node.f_node
                    continue

            # trajectory limit
            if traj_limit_mode:
                traj_len_limit = min(total_len-traj_id, cur_skill.traj_len_limit)
                if traj_len_limit != total_len - traj_id:
                    pdb.set_trace()
            else:
                # pdb.set_trace()
                traj_len_limit = total_len - traj_id
            if traj_len_limit == 0:
                terminated = True
            else:
                if collect_states:
                    if collect_actions:
                        obs, reward, terminated, truncated, info, traj_len, new_states, new_imgs, new_actions = cur_skill.rollout(env, obs, traj_len_limit, collect_states=True, drop_success=True, add_obs_fun=add_obs_fun, collect_states_fun=collect_states_fun, collect_actions=collect_actions)
                        action_store.append([np.expand_dims(action, 0) for action in new_actions])
                    else:
                        obs, reward, terminated, truncated, info, traj_len, new_states, new_imgs = cur_skill.rollout(env, obs, traj_len_limit, collect_states=True, drop_success=True, add_obs_fun=add_obs_fun, collect_states_fun=collect_states_fun)
                    if len(state_store) == 1 and len(state_store[0]) == 1:
                        state_store[-1] += [np.expand_dims(state, 0) for state in new_states]
                        img_store[-1] += new_imgs
                    else:
                        state_store.append([np.expand_dims(state, 0) for state in new_states])
                        img_store.append(new_imgs)
                    # img_store.append(new_imgs)
                    rew_store.append(cur_skill.reward_fun)
                else:
                    obs, reward, terminated, truncated, info, traj_len = cur_skill.rollout(env, obs, traj_len_limit, drop_success=True, add_obs_fun=add_obs_fun)

                # if drop
                if not terminated and traj_len < traj_len_limit:
                    if pick_eval and len(pick_store) > len(success_store):
                        success_store.append(True)
                    success = True
                    task_success = True
                    if env_eval:
                        env_success = True
                    break

                traj_id += traj_len

            task_success = total_len > traj_id and terminated
            if traj_limit_mode:
                success = total_len > traj_id
                if success != task_success:
                    pdb.set_trace()
            else:
                # pdb.set_trace()
                success = task_success
            if pick_eval and len(pick_store) > len(success_store):
                success_store.append(success)

            # success branch
            if success:
                cur_node = cur_node.s_node
            # fail
            else:
                # if collect_states and len(img_store[-1]) != traj_len_limit+1:
                #     pdb.set_trace()
                break

        if env_eval:
            success = env_success
        else:
            success = success and task_success

        if collect_states:
            if pick_eval:
                return obs, info, success, traj_id, state_store, img_store, rew_store, pick_store, success_store, pick_imgs
            else:
                if collect_actions:
                    return obs, info, success, traj_id, state_store, img_store, rew_store, action_store
                return obs, info, success, traj_id, state_store, img_store, rew_store
        if pick_eval:
            return obs, info, success, traj_id, pick_store, success_store, pick_imgs
        else:
            return obs, info, success, traj_id

    # rebuild demo index for nodes in graph
    def rebuild_index(self, obs_seq, thre, cur_node=None):
        # init
        if cur_node is None:
            cur_node = self.start_node
        
        # iterative
        if cur_node.s_node is not None:
            last_ids = self.rebuild_index(obs_seq, thre, cur_node.s_node)
        else:
            last_ids = [len(each_seq) for each_seq in obs_seq]

        # do rebuild
        if cur_node.reward_fun is not None:
            new_ids = []
            rew_fun = cur_node.reward_fun[0]
            for each_obs_seq, last_id in zip(obs_seq, last_ids):
                check = False
                for cur_id in np.arange(last_id-1, -1, -1):
                    # match
                    if not check:
                        if rew_fun.get_reward(np.expand_dims(each_obs_seq[cur_id], 0), 0) >= thre:
                            check = True
                    # match -> unmatch
                    else:
                        if rew_fun.get_reward(np.expand_dims(each_obs_seq[cur_id], 0), 0) < thre:
                            break
                new_ids.append(cur_id+1)
            cur_node.start_idx = new_ids
        
            return new_ids

        return last_ids

    # helper for copy
    def _get_copy(self, old_node, node_dict, skill_drop=False):
        # None
        if old_node is None:
            return None

        # copy children
        new_s_node = self._get_copy(old_node.s_node, node_dict, skill_drop)
        new_f_node = self._get_copy(old_node.f_node, node_dict, skill_drop)

        # copy self
        if old_node.node_id not in node_dict:
            new_node = old_node.get_copy(skill_drop)
            node_dict[old_node.node_id] = new_node
        else:
            new_node = node_dict[old_node.node_id]
        new_node.s_node = new_s_node
        new_node.f_node = new_f_node

        return new_node

    # copy
    def get_copy(self, skill_drop=False):
        node_dict = {}
        new_start_node = self._get_copy(self.start_node, node_dict, skill_drop)
        new_graph = PredicateGraph(None, None, None, new_start_node)
        new_graph.node_num = self.node_num
        new_graph.disable_fail = self.disable_fail

        return new_graph
    
    # helper for store
    def _update_policys(self, old_node, policy_dict):
        # None
        if old_node is None:
            return None
        
        # store
        if old_node.node_id in policy_dict:
            old_node.set_policy(policy_dict[old_node.node_id])

        # store policy for children (TODO)
        self._update_policys(old_node.s_node, policy_dict)
        self._update_policys(old_node.f_node, policy_dict)

        return None

    # store skills
    def update_policys(self, policy_dict):
        self._update_policys(self.start_node, policy_dict)


# iterative program of graph
# use while loop and condition to run graph
class IterativePredicateGraph:
    def __init__(self, iterative_graph, conditions, obs_transit, valid_blocks=None, \
                 pre_iter_graph=None, post_iter_graph=None, reuse_pre_iter_graph=False):
        self.graph = iterative_graph
        self.cond = conditions
        self.obs_transit = obs_transit
        self.valid_blocks = valid_blocks
        # store pre-iteration graph and post-iteration graph
        self.pre_iter_graph = pre_iter_graph
        self.pre_iter_satis = False or self.pre_iter_graph is None
        self.reuse_pre_iter_graph = reuse_pre_iter_graph
        self.post_iter_graph = post_iter_graph
        self.post_iter_satis = False or self.post_iter_graph is None
    
    def load_pre_iter_graph(self, pre_iter_graph):
        self.pre_iter_graph = pre_iter_graph
        self.pre_iter_satis = False

    def get_copy(self):
        # graph
        new_graph = self.graph.get_copy()
        if self.pre_iter_graph is not None:
            pre_iter_graph = self.pre_iter_graph.get_copy()
        if self.post_iter_graph is not None:
            post_iter_graph = self.post_iter_graph.get_copy()
        # condition
        new_cond = {}
        for cond_id in self.cond:
            if self.cond[cond_id] is None:
                new_cond[cond_id] = None
            else:
                new_cond[cond_id] = copy.deepcopy(self.cond[cond_id])
        new_obs_transit = copy.deepcopy(self.obs_transit)
        new_valid_blocks = copy.deepcopy(self.valid_blocks)

        return IterativePredicateGraph(new_graph, new_cond, new_obs_transit, new_valid_blocks, pre_iter_graph, post_iter_graph)

    def print_program(self):
        graph_stru, graph_detail = self.graph.print_graph()
        # TODO: pre and post graph
        if self.pre_iter_graph is not None:
            pre_graph_stru, pre_graph_detail = self.pre_iter_graph.print_graph()
        else:
            pre_graph_stru = ''
            pre_graph_detail = ''

        condition_detail = '\n'.join([str(self.cond[cond_id]) for cond_id in self.cond])
        prog_str = '{}\n while True: \n    find block with {} conditions\n    {}'.format(pre_graph_stru, len(self.cond), graph_stru)

        return prog_str, condition_detail, pre_graph_detail+graph_detail

    def store(self, store_path):
        # store graph architecture
        graph_arch = self.graph.get_copy(skill_drop=True)
        with open(os.path.join(store_path, 'graph.pkl'), 'wb') as f:
            pickle.dump(graph_arch, f)

        if self.pre_iter_graph is not None:
            pre_iter_graph_arch = self.pre_iter_graph.get_copy(skill_drop=True)
            with open(os.path.join(store_path, 'graph.pkl'), 'wb') as f:
                pickle.dump(pre_iter_graph_arch, f)
        if self.post_iter_graph is not None:
            post_iter_graph_arch = self.post_iter_graph.get_copy(skill_drop=True)
            with open(os.path.join(store_path, 'graph.pkl'), 'wb') as f:
                pickle.dump(post_iter_graph_arch, f)

        # store conditions
        with open(os.path.join(store_path, 'conditions.pkl'), 'wb') as f:
            pickle.dump(self.cond, f)


    def eval_condition(self, obs):
        # init
        valid_blocks = self.valid_blocks
        all_boxes_dict = {}
        all_boxes = []
        true_blocks = []

        # find object centric view
        for block_id in valid_blocks:
            # block_obs = np.squeeze(self.obs_transit.do_crop_state(np.expand_dims(obs, 0), block_id), axis=0)
            # b = Box(str(block_id))
            # # TODO: hard code for now
            # b.set_attribute(block_obs[5], block_obs[6], block_obs[7], block_obs[13], block_obs[14], block_obs[15])

            block_obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), block_id, return_dict=True)
            b = Box(str(block_id))
            block_pos = block_obs['block'][0]
            block_goal = block_obs['block_goal'][0]
            b.set_attribute(block_pos[0], block_pos[1], block_pos[2], block_goal[0], block_goal[1], block_goal[2])

            all_boxes_dict[block_id] = b
            all_boxes.append(b)
        # test true block
        for block_id in valid_blocks:
            input = {"target": all_boxes_dict[block_id], 
                     "all_box": all_boxes}
            cond_result = True
            for cond_id in self.cond:
                cond = self.cond[cond_id]
                if cond is not None and not cond.evaluate_specific(input):
                    cond_result = False
                    break
            if cond_result:
                true_blocks.append(block_id)
        
        return true_blocks

    # check status (currently used to set status of pre iteration)
    def check_status(self, start_id):
        if not self.pre_iter_satis:
            parent_node, next_node, task, fail_search = self.pre_iter_graph.get_search_node(start_id)
            if parent_node is None and self.pre_iter_graph.get_spec_node(start_id) is not None:
                self.pre_iter_satis = True

    # get search node based on whether policy has been trained and whether state has terminated 
    def get_search_node(self, start_id=None, state=None):
        # TODO: consider post iteration graph later
        # handle pre iteration graph
        if not self.pre_iter_satis:
            if start_id is not None:
                parent_node, next_node, task, fail_search = self.pre_iter_graph.get_search_node(start_id)
                if parent_node is not None or state is None:
                    return parent_node, next_node, task, fail_search
            else:
                return self.pre_iter_graph.get_search_node()

        # try directly search
        if start_id is not None:
            parent_node, next_node, task, fail_search = self.graph.get_search_node(start_id)
            if parent_node is not None or state is None:
                return parent_node, next_node, task, fail_search
        
        # go back to the beginning of next iteration if not terminate
        if state is not None:
            true_blocks = self.eval_condition(state)
            if len(true_blocks) == 0:
                return None, None, None, None
            else:
                # pick_block_id = np.random.choice(true_blocks, 1).item()
                pick_block_id = true_blocks[0]
                self.graph.set_object_centric(pick_block_id)
                return self.graph.get_search_node()
        
        return None, None, None, None

    # reset
    def do_reset(self, env, **kwargs):
        # reset graph status
        self.pre_iter_satis = False or self.pre_iter_graph is None
        self.post_iter_satis = False or self.post_iter_graph is None

        obs, info = env.reset(**kwargs)

        parent_node, next_node, task, fail_search = self.get_search_node(state=obs)

        return obs, info, parent_node, next_node, task, fail_search

    # loop reset
    def do_loop_reset(self, obs):
        return self.get_search_node(state=obs)

    # rollout pre_iter_graph
    def rollout_pre_iter_graph(self, env, total_len, stop_id, drop_success=False, fail_search=False, traj_limit_mode=False, add_obs_fun=None, **kwargs):
        # already satisfied
        if self.pre_iter_satis:
            return None, None, None, None
            
        obs, info, success, traj_len = self.pre_iter_graph.rollout(env, total_len, stop_id, \
                                                    drop_success=drop_success, fail_search=fail_search, \
                                                    traj_limit_mode=traj_limit_mode, add_obs_fun=add_obs_fun, **kwargs)

        # update satisfied
        self.check_status(self.pre_iter_graph.start_node.node_id)

        return obs, info, success, traj_len

    # rollout
    # full run:
    #       True: run for all while iterations; False: run for only one iteration
    def rollout(self, env, total_len, stop_id, valid_blocks, drop_success=False, fail_search=False, traj_limit_mode=False, start_id=None, exist_obs=None, full_run=False, **kwargs):
        # reset
        if exist_obs is None:
            obs, info = env.reset(**kwargs)
            info['drop'] = False
        else:
            obs = exist_obs
        # init
        traj_limit = total_len
        while traj_limit > 0:
            true_blocks = self.eval_condition(obs)
            # termination
            if len(true_blocks) == 0:
                break
            # pick_block_id = np.random.choice(true_blocks, 1).item()
            pick_block_id = true_blocks[0]
            # rollout for one iteration
            self.graph.set_object_centric(pick_block_id)
            obs, info, success, traj_id = self.graph.rollout(env, traj_limit, stop_id, False, drop_success, fail_search, traj_limit_mode, start_id, obs, **kwargs)
            # check next
            if not success or not full_run:
                return obs, info, success, traj_id
            traj_limit = traj_limit - traj_id

    # rollout for evaluation
    def eval_rollout(self, env, total_len, valid_blocks, collect_states=False, pick_eval=False, env_eval=False, traj_limit_mode=False, **kwargs):
        # not support pick eval for now
        assert not pick_eval
        # reset
        obs, info = env.reset(**kwargs)
        # init
        if collect_states:
            state_store = []
            img_store = []
            rew_store = []
        traj_limit = total_len
        while traj_limit > 0:
            # find object id
            true_blocks = []
            all_boxes = []
            all_boxes_dict = {}
            for block_id in valid_blocks:
                # block_obs = np.squeeze(self.obs_transit.do_crop_state(np.expand_dims(obs, 0), block_id), axis=0)
                # b = Box(str(block_id))
                # b.set_attribute(block_obs[3], block_obs[4], block_obs[5], block_obs[9], block_obs[10], block_obs[11])

                block_obs = self.obs_transit.do_crop_state(np.expand_dims(obs, 0), block_id, return_dict=True)
                b = Box(str(block_id))
                block_pos = block_obs['block'][0]
                block_goal = block_obs['block_goal'][0]
                b.set_attribute(block_pos[0], block_pos[1], block_pos[2], block_goal[0], block_goal[1], block_goal[2])

                all_boxes_dict[block_id] = b
                all_boxes.append(b)
            # test true block
            for block_id in valid_blocks:
                input = {"target": all_boxes_dict[block_id], 
                         "all_box": all_boxes}
                cond_result = True
                for cond_id in self.cond:
                    cond = self.cond[cond_id]
                    if cond is not None and not cond.evaluate_specific(input):
                        cond_result = False
                        break
                if cond_result:
                    true_blocks.append(block_id)
            # termination
            if len(true_blocks) == 0:
                break
            # pick_block_id = np.random.choice(true_blocks, 1).item()
            pick_block_id = true_blocks[0]
            # rollout for one iteration
            self.graph.set_object_centric(pick_block_id)
            results = self.graph.eval_rollout(env, total_len, collect_states=collect_states, 
                                            pick_eval=pick_eval, env_eval=env_eval, traj_limit_mode=traj_limit_mode, 
                                            exist_obs=obs, **kwargs)
            if collect_states:
                state_store += results[4]
                img_store += results[5]
                rew_store += results[6]
            obs, info, success, traj_id = results[:4]

            # check next
            if not success:
                if collect_states:
                    return obs, info, success, traj_id, state_store, img_store, rew_store
                return obs, info, success, traj_id
            traj_limit = traj_limit - traj_id
        
        # return evaluation results
        if collect_states:
            return obs, info, success, traj_id, state_store, img_store, rew_store
        return obs, info, success, traj_id