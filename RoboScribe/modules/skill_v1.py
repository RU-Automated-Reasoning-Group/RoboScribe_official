import pdb
import numpy as np
import copy
import math
from modules.reward import RewIdentity

class Skill:
    def __init__(self, policy, reward_fun, done_thre, hold_len=1, lagrangian=False, traj_len_limit=math.inf):
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

    def reset(self):
        pass

    def get_copy(self):
        return Skill(copy.deepcopy(self.policy), self.reward_fun.get_copy(), self.done_thre, self.hold_len)

    def rollout(self, env, obs, skill_limit, collect_states=False, drop_success=False):
        # debug
        if not isinstance(self.reward_fun, RewIdentity):
            assert not self.reward_fun.train

        # init
        rollout_num = 0
        hold_step = 0
        if collect_states:
            state_list = []
            img_list = []

        for frame_id in range(skill_limit):
            action, _ = self.policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # drop if environment success
            if drop_success and reward == 0:
                terminated = truncated = False
                if collect_states:
                    state_list.append(obs)
                    img_list.append(env.render())
                break

            reward = self.reward_fun.get_reward(obs, reward)
            if reward >= self.done_thre:
                hold_step += 1
            else:
                hold_step = 0

            if collect_states:
                state_list.append(obs)
                img_list.append(env.render())
            rollout_num += 1
            if hold_step >= self.hold_len:
                break
        
        # terminate if success
        if hold_step >= self.hold_len:
            terminated = truncated = True
        
        if collect_states:
            return obs, reward, terminated, truncated, info, rollout_num, state_list, img_list
        return obs, reward, terminated, truncated, info, rollout_num
    
# predicate node (state node and out edge policy)
class PredicateNode:
    def __init__(self, reward_fun, start_idx, node_id, skill=None, s_node=None, f_node=None, traj_len_limit=None):
        # node structure
        self.s_node = s_node
        self.f_node = f_node

        # others
        self.reward_fun = reward_fun
        self.start_idx = start_idx
        self.node_id = node_id

        # skill
        self.skill = skill
        self.traj_len_limit = traj_len_limit

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
        # copy skill
        if self.skill is None or skill_drop:
            new_skill = None
        else:
            new_skill = copy.deepcopy(self.skill)
        # new node
        new_node = PredicateNode(new_reward_fun, copy.deepcopy(self.start_idx), \
                                 self.node_id, skill=new_skill, traj_len_limit=self.traj_len_limit)

        return new_node

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
                fail_check = True if fail_check is None else fail_check
                return parent_node, fail_check
            
        # otherwise
        return None, None

    # get the next node for skill train
    def get_search_node(self):        
        parent_node, fail_search = self._get_search_node(self.start_node)
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
        visited_ids = [False for _ in range(self.node_num)]
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

    # add node containing new reward function
    def add_node(self, new_rew_fun, new_start_id, parent_node, target_node):
        new_node = PredicateNode(new_rew_fun, new_start_id, node_id=self.node_num)
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
    def rollout(self, env, total_len, stop_id, collect_states=False, drop_success=False, fail_search=False, **kwargs):
        # reset
        obs, info = env.reset(**kwargs)
        info['drop'] = False
        traj_id = 0
        state_store = []
        img_store = []    

        # rollout skill graph
        success = True
        collect_success = None
        head_node = self.find_head_fail(stop_id)
        cur_node = self.start_node
        while cur_node is not None and cur_node.node_id != stop_id:
            cur_skill = cur_node.skill
            if cur_skill is None:
                pdb.set_trace()
                assert isinstance(cur_node.reward_fun[-1], RewIdentity)
                break
            cur_skill.reset()

            # trajectory limit
            if cur_skill.traj_len_limit > 0:
                if cur_node.f_node is not None:
                    traj_len_limit = min(total_len - traj_id, cur_skill.traj_len_limit)
                else:
                    traj_len_limit = total_len - traj_id
                if collect_states:
                    obs, reward, terminated, truncated, info, traj_len, new_states, new_imgs = cur_skill.rollout(env, obs, traj_len_limit, collect_states=True, drop_success=drop_success)
                    state_store += [np.expand_dims(state, 0) for state in new_states]
                    img_store += new_imgs
                else:
                    obs, reward, terminated, truncated, info, traj_len = cur_skill.rollout(env, obs, traj_len_limit, drop_success=drop_success)
                traj_id += traj_len
            # drop success
            info['drop'] = False
            if drop_success and traj_len < traj_len_limit and not terminated:
                info['drop'] = True
                success = False
                collect_success = False
                break

            success = total_len > traj_id and terminated
            # success branch
            if success:
                if fail_search and cur_node.node_id == head_node.node_id:
                    success = False
                    collect_success = False
                    break
                cur_node = cur_node.s_node
                if cur_node.node_id == stop_id:
                    collect_success = False
            # fail branch
            elif fail_search:
                if cur_node.f_node is None:
                    if cur_node.s_node.node_id == stop_id:
                        collect_success = True
                    else:
                        collect_success = False
                    break
                elif cur_node.f_node.node_id != stop_id and cur_node.f_node.skill is None:
                    collect_success = False
                    break
                elif total_len > traj_id:
                    success = True
                    cur_node = cur_node.f_node
                    if cur_node.node_id == stop_id:
                        collect_success = True
                else:
                    if cur_node.s_node is not None and cur_node.s_node.node_id == stop_id:
                        collect_success = True
                    else:
                        collect_success = False
                    break
            # fail
            else:
                # Whether collect previous fail
                if cur_node.s_node.node_id == stop_id:
                    collect_success = True
                else:
                    collect_success = False
                break

        if self.start_node.node_id != stop_id:
            assert collect_success is not None
        if collect_states:
            return obs, info, success, collect_success, traj_id, state_store, img_store
        return obs, info, success, traj_id

    # rollout for evaluation
    def eval_rollout(self, env, total_len, collect_states=False, **kwargs):
        # reset
        obs, info = env.reset(**kwargs)
        traj_id = 0
        state_store = []
        img_store = []
        rew_store = []

        # rollout skill graph
        success = True
        cur_node = self.start_node
        while cur_node.skill is not None:
            cur_skill = cur_node.skill
            if cur_skill is None:
                pdb.set_trace()
                assert isinstance(cur_node.reward_fun[-1], RewIdentity)
                break
            cur_skill.reset()

            # trajectory limit
            if cur_node.f_node is not None and not self.disable_fail:
                traj_len_limit = min(total_len - traj_id, cur_skill.traj_len_limit)
            else:
                traj_len_limit = total_len - traj_id
            if traj_len_limit == 0:
                terminated = True
            else:
                if collect_states:
                    obs, reward, terminated, truncated, info, traj_len, new_states, new_imgs = cur_skill.rollout(env, obs, traj_len_limit, collect_states=True, drop_success=True)
                    state_store.append([np.expand_dims(state, 0) for state in new_states])
                    img_store.append(new_imgs)
                    rew_store.append(cur_skill.reward_fun)
                else:
                    obs, reward, terminated, truncated, info, traj_len = cur_skill.rollout(env, obs, traj_len_limit, drop_success=True)

                # if drop
                if not terminated and traj_len < traj_len_limit:
                    success = True
                    break

                traj_id += traj_len

            success = total_len > traj_id and terminated
            # success branch
            if success:
                cur_node = cur_node.s_node
            # fail branch
            elif cur_node.f_node is not None and not self.disable_fail:
                cur_node = cur_node.f_node
            # fail
            else:
                if len(img_store[-1]) != traj_len_limit:
                    pdb.set_trace()
                break

        if collect_states:
            return obs, info, success, traj_id, state_store, img_store, rew_store
        return obs, info, success, traj_id


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