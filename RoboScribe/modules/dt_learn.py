from sklearn import tree
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import random
from tqdm import tqdm
import copy
import re
import math
import matplotlib.pyplot as plt

import pdb

class DenseCls:
    def __init__(self, rules, eps, obs_transit, only_old=False):
        # for rule
        self.rules = rules
        self.eps = eps
        self.dense_rules = self.get_dense(self.rules)
        self.dense_details = self.get_dense_detail(self.rules)
        self.comb_rule = ' or '.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.rules])
        self.comb_dense_rule = self.comb_dense(self.dense_rules)
        # for others
        self.obs_transit = obs_transit
        self.only_old = only_old
        self.old_rules = None
        self.old_comb_rule = None

    def set_new_rules(self, new_rules):
        self.rules = new_rules
        self.dense_rules = self.get_dense(self.rules)
        self.dense_details = self.get_dense_detail(self.rules)
        self.comb_rule = ' or '.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.rules])
        self.comb_dense_rule = self.comb_dense(self.dense_rules)

    def get_dense(self, rules):
        raise NotImplementedError

    def comb_dense(self, dense_rules):
        raise NotImplementedError

    def get_dense_detail(self, rules):
        raise NotImplementedError

    def execute(self, obs):
        if self.obs_transit is not None:
            obs = self.obs_transit.get_abs_obs(obs)
        obs = np.squeeze(obs, axis=0)
        reward = -eval(self.comb_dense_rule)

        return reward
    
    def execute_dense_details(self, obs):
        if self.obs_transit is not None:
            obs = self.obs_transit.get_abs_obs(obs)
        obs = np.squeeze(obs, axis=0)

        dense_results = []
        self.dense_details = self.get_dense_detail(self.rules)
        for dense_rules in self.dense_details:
            dense_results.append([])
            for rule in dense_rules:
                dense_results[-1].append(-eval(rule))

        return dense_results

    def execute_rules(self, obs):
        if self.obs_transit is not None:
            obs = self.obs_transit.get_abs_obs(obs)
        obs = np.squeeze(obs, axis=0)
        bool_res = eval(self.comb_rule)

        return bool_res

    def execute_details(self, obs):
        if self.obs_transit is not None:
            obs = self.obs_transit.get_abs_obs(obs)
        obs = np.squeeze(obs, axis=0)

        results = []
        for rule in self.dense_rules:
            results.append(-eval(rule))

        return results

    def adjust_rules(self, all_obs):
        # init
        all_thresholds = []
        for rule in self.rules:
            new_threshold = {}
            # init values
            for term in rule:
                terms = re.split('>|<=', term)
                if terms[0] not in new_threshold:
                    new_threshold[terms[0]] = [math.inf, -math.inf]
                if '>' in term:
                    new_threshold[terms[0]][0] = float(terms[1])
                elif '<=' in term:
                    new_threshold[terms[0]][1] = float(terms[1])
            # add
            all_thresholds.append(new_threshold)

        # calculate threshold
        for obs in all_obs:
            # pick a rule
            dense_results = self.execute_details(np.expand_dims(obs, 0))
            if self.obs_transit is not None:
                obs = self.obs_transit.get_abs_obs(np.expand_dims(obs, 0))
                obs = np.squeeze(obs, axis=0)
            rule_id = np.argmax(dense_results)
            pick_rule = self.rules[rule_id]
            new_threshold = all_thresholds[rule_id]
            # get new threshold
            for term in pick_rule:
                terms = re.split('>|<=', term)
                if '>' in term and eval(terms[0]) <= new_threshold[terms[0]][0]:
                    new_threshold[terms[0]][0] = eval(terms[0])-1e-5
                elif '<=' in term and eval(terms[0]) > new_threshold[terms[0]][1]:
                    new_threshold[terms[0]][1] = eval(terms[0])

        # renew rules
        new_rules = []
        for rule_id, rule in enumerate(self.rules):
            # skip if not need
            if len(all_thresholds[rule_id]) == 0:
                new_rules.append(rule)
                continue
            # update each term
            new_rules.append([])
            new_threshold = all_thresholds[rule_id]
            for term in rule:
                terms = re.split('>|<=', term)
                if terms[0] not in new_threshold:
                    new_rules[-1].append(terms)
                    continue
                if '>' in term:
                    terms[1] = str(new_threshold[terms[0]][0])
                    new_rules[-1].append(terms[0]+'>'+terms[1])
                elif '<=' in term:
                    terms[1] = str(new_threshold[terms[0]][1])
                    new_rules[-1].append(terms[0]+'<='+terms[1])

        # renew all
        self.rules = new_rules
        self.dense_rules = self.get_dense(self.rules)
        self.comb_rule = ' or '.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.rules])
        self.comb_dense_rule = self.comb_dense(self.dense_rules)

    def opt_threshold_back(self, all_obs):
        # init
        all_thresholds = []
        all_opt_thresholds = []
        for rule in self.rules:
            new_threshold = {}
            new_opt_threshold = {}
            # init values
            for term in rule:
                terms = re.split('>|<=', term)
                if terms[0] not in new_threshold:
                    new_threshold[terms[0]] = [-math.inf, math.inf]
                if '>' in term:
                    new_threshold[terms[0]][0] = float(terms[1])
                elif '<=' in term:
                    new_threshold[terms[0]][1] = float(terms[1])
                new_opt_threshold[terms[0]] = []
            # add
            all_thresholds.append(new_threshold)
            all_opt_thresholds.append(new_opt_threshold)

        # gather observations
        for obs in all_obs:
            results = self.execute_details(np.expand_dims(obs, axis=0))
            if self.obs_transit is not None:
                obs = self.obs_transit.get_abs_obs(np.expand_dims(obs, axis=0))
            obs = np.squeeze(obs, axis=0)
            for r_id, r in enumerate(results):
                if not r:
                    continue
                cur_threshold = all_opt_thresholds[r_id]
                for term in cur_threshold:
                    cur_threshold[term].append(eval(term))

        # do optimize
        new_rules = []
        for threshold, threshold_val in zip(all_thresholds, all_opt_thresholds):
            rule = []
            for term in threshold:
                if threshold[term][0] != -math.inf:
                    rule.append(term + '>' + str(min(threshold_val[term])))
                if threshold[term][1] != math.inf:
                    rule.append(term + '<=' + str(max(threshold_val[term])))
            new_rules.append(rule)

        # renew all
        self.rules = new_rules
        self.dense_rules = self.get_dense(self.rules)
        self.comb_rule = ' or '.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.rules])
        self.comb_dense_rule = self.comb_dense(self.dense_rules)

    def opt_threshold(self, all_obs, alpha=1.0, eps=1e-5):
        # init
        all_thresholds = []
        all_opt_thresholds = []
        for rule in self.rules:
            new_threshold = {}
            new_opt_threshold = {}
            # init values
            for term in rule:
                terms = re.split('>|<=', term)
                if terms[0] not in new_threshold:
                    new_threshold[terms[0]] = [-math.inf, math.inf]
                if '>' in term:
                    new_threshold[terms[0]][0] = float(terms[1])
                elif '<=' in term:
                    new_threshold[terms[0]][1] = float(terms[1])
                new_opt_threshold[terms[0]] = []
            # add
            all_thresholds.append(new_threshold)
            all_opt_thresholds.append(new_opt_threshold)

        # gather observations
        for obs in all_obs:
            results = self.execute_details(np.expand_dims(obs, axis=0))
            if self.obs_transit is not None:
                obs = self.obs_transit.get_abs_obs(np.expand_dims(obs, axis=0))
            obs = np.squeeze(obs, axis=0)
            for r_id, r in enumerate(results):
                # if r < 0:
                #     continue
                if r < -eps:
                    continue
                cur_threshold = all_opt_thresholds[r_id]
                for term in cur_threshold:
                    cur_threshold[term].append(eval(term))

        # do optimize
        new_rules = []
        for threshold, threshold_val in zip(all_thresholds, all_opt_thresholds):
            rule = []
            for term in threshold:
                if threshold[term][0] != -math.inf:
                    val = max([threshold[term][0], min(threshold_val[term])])
                    val = alpha * val + (1-alpha) * threshold[term][0]
                    rule.append(term + '>' + str(val))
                if threshold[term][1] != math.inf:
                    val = min([threshold[term][1], max(threshold_val[term])])
                    val = alpha * val + (1-alpha) * threshold[term][1]
                    rule.append(term + '<=' + str(val))
            new_rules.append(rule)

        # renew all
        self.rules = new_rules
        self.dense_rules = self.get_dense(self.rules)
        self.comb_rule = ' or '.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.rules])
        self.comb_dense_rule = self.comb_dense(self.dense_rules)

    def _do_and_comb(self, rules1, rules2):
        # init
        rules_threshold = {}

        # get value
        for rules in [rules1, rules2]:
            for term in rules:
                terms = re.split('>|<=', term)
                if terms[0] not in rules_threshold:
                    rules_threshold[terms[0]] = [-math.inf, math.inf]
                if '>' in term and float(terms[1]) > rules_threshold[terms[0]][0]:
                    rules_threshold[terms[0]][0] = float(terms[1])
                elif '<=' in term and float(terms[1]) < rules_threshold[terms[0]][1]:
                    rules_threshold[terms[0]][1] = float(terms[1])

        # get rule
        new_rules = []
        for state in rules_threshold:
            range = rules_threshold[state]
            # invalid
            if range[0] >= range[1]:
                return []
            # otherwise
            if range[0] != -math.inf:
                new_rules.append('{}>{}'.format(state, range[0]-1e-5))
            if range[1] != math.inf:
                new_rules.append('{}<={}'.format(state, range[1]))

        return new_rules

    def comb_rules(self, new_rules, comb_method):
        # skip
        if len(new_rules) == 0:
            return

        if comb_method == 'and':
            self.comb_and(new_rules)
        else:
            raise NotImplementedError

    # combine with and operator
    def comb_and(self, new_rules):
        result_rules = []
        for new_rule in new_rules:
            for exist_rule in self.rules:
                result = self._do_and_comb(new_rule, exist_rule)
                if len(result) > 0:
                    result_rules.append(result)

        if len(result_rules) == 0:
            print('what?')
            pdb.set_trace()

        # store old
        if self.only_old:
            self.old_rules = self.rules
            self.old_comb_rule = self.comb_rule
        # renew all
        self.rules = result_rules
        self.dense_rules = self.get_dense(self.rules)
        self.comb_rule = ' or '.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.rules])
        self.comb_dense_rule = self.comb_dense(self.dense_rules)

    # prune or rule
    def prune_rules(self, positive_states, min_sample):
        if len(self.rules) <= 1:
            return True
        results = np.zeros((len(self.rules), len(positive_states)))
        for p_s_id, p_s in enumerate(positive_states):
            res = self.execute_details(np.expand_dims(p_s, 0))
            if self.lagrangian_rules is not None:
                results[:, p_s_id] = np.asarray(res[:-len(self.lagrangian_rules)])
            else:
                results[:, p_s_id] = np.asarray(res)

        results = np.sum(results, axis=0)
        new_rules = []
        for res_id, res in enumerate(results):
            if res > min_sample:
                new_rules.append(self.rules[res_id])

        self.set_new_rules(new_rules)

    # prune and rule
    def simplify_rules(self, negative_states):
        # get dense results
        results = [np.zeros((len(self.rules[rule_id]), len(negative_states))) for rule_id in range(len(self.rules))]
        for n_s_id, n_s in enumerate(negative_states):
            res = self.execute_dense_details(np.expand_dims(n_s, 0))
            for rule_id in range(len(self.rules)):
                results[rule_id][:, n_s_id] = np.array(res[rule_id]) > 0

        # from simple to complex for each or rule
        new_rules = []
        for rule_id in range(len(self.rules)):
            if len(self.rules[rule_id]) == 0:
                continue
            # check the first array
            finish = False
            goal_result = np.sum(results[rule_id], axis=0) == len(self.rules[rule_id])
            for dense_id in range(len(self.rules[rule_id])):
                if np.sum(results[rule_id][dense_id]) == np.sum(goal_result):
                    new_rules.append([self.rules[dense_id]])
                    break
            if finish:
                continue

            pdb.set_trace()

            # init
            check_dict = {dense_id:[results[rule_id][dense_id].copy()] for dense_id in range(len(self.rules[rule_id]))}
            contain_dict = {dense_id:[[dense_id]] for dense_id in range(len(self.rules[rule_id]))}
            for dense_num in range(2, len(self.rules[rule_id])+1):
                new_check_dict = {}
                new_contain_dict = {}
                for prev_id in check_dict:
                    for next_id in range(prev_id+1, len(self.rules[rule_id])):
                        if next_id not in new_check_dict:
                            new_check_dict[next_id] = []
                            new_contain_dict[next_id] = []
                        # check the new one
                        for prev_check_array, prev_contain_array in zip(check_dict[prev_id], contain_dict[prev_id]):
                            new_check_array = (prev_check_array+results[rule_id][next_id]) == 2
                            new_contain_array = prev_contain_array + [next_id]
                            if np.sum(new_check_array) == np.sum(goal_result):
                                finish = True
                                new_rules.append([self.rules[rule_id][contain_id] for contain_id in new_contain_array])
                                break
                            new_check_dict[next_id].append(new_check_array)
                            new_contain_dict[next_id].append(new_contain_array)
                        # complete
                        if finish:
                            break
                    if finish:
                        break
                if finish:
                    break
                pdb.set_trace()
                check_dict = new_check_dict
                contain_dict = new_contain_dict

            # guarentee to finish
            assert finish

        pdb.set_trace()
        self.set_new_rules(new_rules)

    def split_predicate(self):
        all_cls = [DenseCls([rule], self.eps, self.obs_transit) for rule in self.rules]
        return all_cls

    def get_copy(self):
        raise NotImplementedError

    def __str__(self):
        print_rule = ' or\n'.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.rules])
        print_dense_rule = ' *\n'.join(self.dense_rules)

        return print_rule + '\n\n' + print_dense_rule


class TreeCutDenseCls(DenseCls):
    def __init__(self, rules, eps=1e-5, obs_transit=None, only_old=False):
        super().__init__(rules, eps, obs_transit, only_old)

    def get_dense(self, rules):
        dense_rules = []
        for rule in rules:
            # get dense
            comb_term = []
            for term in rule:
                if '<=' in term:
                    terms = term.split('<=')
                    d_term = 'max({}-{}, 0)'.format(terms[0], terms[1])
                else:
                    assert '>' in term
                    terms = term.split('>')
                    d_term = 'max({}+{}-{}, 0)'.format(terms[1], self.eps, terms[0])
                comb_term.append(d_term)
            # store
            dense_rules.append('( ' + ' + '.join(comb_term) + ' )')
        
        return dense_rules

    def comb_dense(self, dense_rules):
        return ' * '.join(dense_rules)


# obs < c: obs - c
# r1 and r2: max(r1, r2)
# r1 or r2: min(r1, r2)
class TreeContDenseCls(DenseCls):
    def __init__(self, rules, eps=1e-5, obs_transit=None, only_old=False):
        super().__init__(rules, eps, obs_transit, only_old)

    def get_dense(self, rules):
        dense_rules = []
        for rule in rules:
            # get dense
            comb_term = []
            for term in rule:
                if '<=' in term:
                    terms = term.split('<=')
                    d_term = '{}-{}'.format(terms[0], terms[1])
                    # d_term = 'max({}-{}, 0)'.format(terms[0], terms[1])
                else:
                    assert '>' in term
                    terms = term.split('>')
                    d_term = '{}+{}-{}'.format(terms[1], self.eps, terms[0])
                    # d_term = 'max({}+{}-{}, 0)'.format(terms[1], self.eps, terms[0])
                comb_term.append(d_term)
            # store
            dense_rules.append('max([ ' + ' , '.join(comb_term) + ' ])')
            # dense_rules.append('( ' + ' + '.join(comb_term) + ' )')
        
        return dense_rules
    
    def execute_rules(self, obs):
        if not self.only_old or self.old_comb_rule is None:
            return super().execute_rules(obs)
        else:
            if self.obs_transit is not None:
                obs = self.obs_transit.get_abs_obs(obs)
            obs = np.squeeze(obs, axis=0)
            bool_res = eval(self.old_comb_rule)
            return bool_res

    # currently for debug
    def get_dense_detail(self, rules):
        dense_rules = []
        for rule in rules:
            # get dense
            comb_term = []
            for term in rule:
                if '<=' in term:
                    terms = term.split('<=')
                    d_term = '{}-{}'.format(terms[0], terms[1])
                else:
                    assert '>' in term
                    terms = term.split('>')
                    d_term = '{}+{}-{}'.format(terms[1], self.eps, terms[0])
                comb_term.append(d_term)
            # store
            dense_rules.append(comb_term)
        
        return dense_rules

    def get_dense_complex(self, rules):
        dense_rules = []
        for rule in rules:
            # get dense
            comb_term = {}
            for term in rule:
                if '<=' in term:
                    terms = term.split('<=')
                    d_term = '{}-{}'.format(terms[0], terms[1])
                else:
                    assert '>' in term
                    terms = term.split('>')
                    d_term = '{}+{}-{}'.format(terms[1], self.eps, terms[0])
                if terms[0] not in comb_term:
                    comb_term[terms[0]] = []
                comb_term[terms[0]].append(d_term)
            # store
            new_rule = ''
            for term_name in comb_term:
                if len(comb_term[term_name]) == 1:
                    new_rule += comb_term[term_name][0] + ' + '
                else:
                    new_rule += 'max([ ' + ' , '.join(comb_term[term_name]) + ' ])' + ' + '

            new_rule = new_rule[:-3]
            dense_rules.append(new_rule)
        
        return dense_rules

    def get_copy(self):
        return TreeContDenseCls(self.rules, self.eps, copy.deepcopy(self.obs_transit))

    def comb_dense(self, dense_rules):
        return 'min([ ' + ' , '.join(dense_rules) + ' ])'
        # return ' * '.join(self.dense_rules)

    def split_predicate(self, samples=None):
        # extract rules
        all_cls = [TreeContDenseCls([rule], self.eps, self.obs_transit) for rule in self.rules]
        
        # sort rules based on success rate on samples
        success_rates = [0 for _ in all_cls]
        for sample in samples:
            for cls_id, each_cls in enumerate(all_cls):
                success_rates[cls_id] += int(each_cls.execute_rules(np.expand_dims(sample, axis=0)))

        all_cls = [each_cls for _, each_cls in sorted(zip(success_rates, all_cls), key=lambda x: x[0], reverse=True)]

        return all_cls

# lagrangian
class TreeContDenseLagrangianCls(TreeContDenseCls):
    def __init__(self, rules, eps=1e-5, obs_transit=None, only_old=False):
        super().__init__(rules, eps, obs_transit, only_old)
        self.lagrangian_rules = None
        self.non_lagrangian_rules = None

    def lag_not_exist(self):
        return self.lagrangian_rules is None

    def set_new_rules(self, new_rules, new_lagrangian_rules, new_non_lagrangian_rules):
        self.rules = new_rules
        self.dense_rules = self.get_dense(self.rules)
        self.dense_details = self.get_dense_detail(self.rules)
        self.comb_rule = ' or '.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.rules])
        self.comb_dense_rule = self.comb_dense(self.dense_rules)

        if new_lagrangian_rules is not None and len(new_lagrangian_rules) != 0:
            self.lagrangian_rules = new_lagrangian_rules
            self.lagrangian_dense_rules = self.get_dense(self.lagrangian_rules)
            self.lagrangian_dense_details = self.get_dense_detail(self.lagrangian_rules)
            self.lagrangian_comb_rule = ' or '.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.lagrangian_rules])
            self.lagrangian_comb_dense_rule = self.comb_dense(self.lagrangian_dense_rules)

        if new_non_lagrangian_rules is not None and len(new_non_lagrangian_rules) != 0:
            self.non_lagrangian_rules = new_non_lagrangian_rules
            self.non_lagrangian_dense_rules = self.get_dense(self.non_lagrangian_rules)
            self.non_lagrangian_dense_details = self.get_dense_detail(self.non_lagrangian_rules)
            self.non_lagrangian_comb_rule = ' or '.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.non_lagrangian_rules])
            self.non_lagrangian_comb_dense_rule = self.comb_dense(self.non_lagrangian_dense_rules)

    def get_complete_rules(self):
        # return origin
        if self.lagrangian_rules is None:
            return self.rules
        # return combined
        result_rules = []
        for new_rule in self.lagrangian_rules:
            for exist_rule in self.rules:
                result = self._do_and_comb(new_rule, exist_rule)
                if len(result) > 0:
                    result_rules.append(result)

        return result_rules

    def comb_rules(self, new_rules, comb_method):
        # skip
        if len(new_rules) == 0:
            return

        if comb_method == 'and':
            self.comb_and(new_rules)
        elif comb_method == 'lagrangian':
            self.comb_lagrangian(new_rules)
        elif comb_method == 'non_lagrangian':
            self.comb_non_lagrangian(new_rules)
        else:
            raise NotImplementedError
    
    def comb_lagrangian(self, new_rules):
        # store rules
        self.lagrangian_rules = copy.deepcopy(new_rules)
        self.lagrangian_dense_rules = self.get_dense(self.lagrangian_rules)
        self.lagrangian_dense_details = self.get_dense_detail(self.lagrangian_rules)
        self.lagrangian_comb_rule = ' or '.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.lagrangian_rules])
        self.lagrangian_comb_dense_rule = self.comb_dense(self.lagrangian_dense_rules)

    def comb_non_lagrangian(self, new_rules):
        # store rules
        self.non_lagrangian_rules = copy.deepcopy(new_rules)
        self.non_lagrangian_dense_rules = self.get_dense(self.non_lagrangian_rules)
        self.non_lagrangian_dense_details = self.get_dense_detail(self.non_lagrangian_rules)
        self.non_lagrangian_comb_rule = ' or '.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.non_lagrangian_rules])
        self.non_lagrangian_comb_dense_rule = self.comb_dense(self.non_lagrangian_dense_rules)

    # prune or rule
    def prune_rules(self, positive_states, min_sample):
        if len(self.rules) <= 1:
            return True
        results = np.zeros((len(self.rules), len(positive_states)))
        for p_s_id, p_s in enumerate(positive_states):
            res = self.execute_details(np.expand_dims(p_s, 0))
            if self.lagrangian_rules is not None:
                results[:, p_s_id] = np.asarray(res[:-len(self.lagrangian_rules)])
            else:
                results[:, p_s_id] = np.asarray(res)

        # TODO: consider eps=-1e-5
        results = np.sum(results>0, axis=-1)

        new_rules = []
        for res_id, res in enumerate(results):
            if res > min_sample:
                new_rules.append(self.rules[res_id])

        self.set_new_rules(new_rules, self.lagrangian_rules, self.non_lagrangian_rules)

    # prune and rule
    def simplify_rules(self, negative_states):
        print('need further debug')
        pdb.set_trace()

        # get dense results
        results = [np.zeros((len(self.rules[rule_id]), len(negative_states))) for rule_id in range(len(self.rules))]
        for n_s_id, n_s in enumerate(negative_states):
            res = self.execute_dense_details(np.expand_dims(n_s, 0))
            for rule_id in range(len(self.rules)):
                results[rule_id][:, n_s_id] = np.array(res[rule_id]) > 0

        # from simple to complex for each or rule
        new_rules = []
        for rule_id in range(len(self.rules)):
            if len(self.rules[rule_id]) == 0:
                continue
            # check the first array
            finish = False
            goal_result = np.sum(results[rule_id], axis=0) == len(self.rules[rule_id])
            for dense_id in range(len(self.rules[rule_id])):
                if np.sum(results[rule_id][dense_id]) == np.sum(goal_result):
                    new_rules.append([self.rules[dense_id]])
                    break
            if finish:
                continue

            pdb.set_trace()

            # init
            check_dict = {dense_id:[results[rule_id][dense_id].copy()] for dense_id in range(len(self.rules[rule_id]))}
            contain_dict = {dense_id:[[dense_id]] for dense_id in range(len(self.rules[rule_id]))}
            for dense_num in range(2, len(self.rules[rule_id])+1):
                new_check_dict = {}
                new_contain_dict = {}
                for prev_id in check_dict:
                    for next_id in range(prev_id+1, len(self.rules[rule_id])):
                        if next_id not in new_check_dict:
                            new_check_dict[next_id] = []
                            new_contain_dict[next_id] = []
                        # check the new one
                        for prev_check_array, prev_contain_array in zip(check_dict[prev_id], contain_dict[prev_id]):
                            new_check_array = (prev_check_array+results[rule_id][next_id]) == 2
                            new_contain_array = prev_contain_array + [next_id]
                            if np.sum(new_check_array) == np.sum(goal_result):
                                finish = True
                                new_rules.append([self.rules[rule_id][contain_id] for contain_id in new_contain_array])
                                break
                            new_check_dict[next_id].append(new_check_array)
                            new_contain_dict[next_id].append(new_contain_array)
                        # complete
                        if finish:
                            break
                    if finish:
                        break
                if finish:
                    break
                pdb.set_trace()
                check_dict = new_check_dict
                contain_dict = new_contain_dict

            # guarentee to finish
            assert finish

        pdb.set_trace()
        self.set_new_rules(new_rules, self.lagrangian_rules, self.non_lagrangian_rules)

    def check_match(self, comp_rules):
        if len(comp_rules) != len(self.rules):
            return False
        for comp_rule, self_rule in zip(comp_rules, self.rules):
            if len(comp_rule) != len(self_rule):
                return False
            for term1, term2 in zip(comp_rule, self_rule):
                if ('<=' in term1) != ('<=' in term2):
                    return False
                
                ori_id = int(term1[term1.find('[')+1 : term1.find(']')])
                comp_id = int(term2[term2.find('[')+1 : term2.find(']')])
                if not self.obs_transit.check_match_ids(ori_id, comp_id):
                    return False
        
        return True

    def _switch_helper(self, term, obj_id, new_obs_transit=None):
        new_term = copy.deepcopy(term)
        left_end = new_term.find('[')
        right_end = new_term.find(']')
        if new_obs_transit is None:
            goal_id = self.obs_transit.get_switch_ids(int(new_term[left_end+1 : right_end]), obj_id)
        else:
            goal_id = self.obs_transit.get_switch_ids_diff_env(int(new_term[left_end+1 : right_end]), obj_id, new_obs_transit.check_list)
        new_term = new_term[:left_end+1] + str(goal_id) + new_term[right_end:]

        return new_term

    def switch_match(self, obj_id, replace=False):
        new_rules = []
        for self_rule in self.rules:
            new_rules.append([])
            for term in self_rule:
                new_term = self._switch_helper(term, obj_id)
                new_rules[-1].append(new_term)
        
        new_lagrangian_rules = []
        if self.lagrangian_rules is not None:
            for self_rule in self.lagrangian_rules:
                new_lagrangian_rules.append([])
                for term in self_rule:
                    new_term = self._switch_helper(term, obj_id)
                    new_lagrangian_rules[-1].append(new_term)

        new_non_lagrangian_rules = []
        if self.non_lagrangian_rules is not None:
            for self_rule in self.non_lagrangian_rules:
                new_non_lagrangian_rules.append([])
                for term in self_rule:
                    new_term = self._switch_helper(term, obj_id)
                    new_non_lagrangian_rules[-1].append(new_term)

        if replace:
            self.set_new_rules(new_rules, new_lagrangian_rules, new_non_lagrangian_rules)

        return new_rules
    
    # switch a different environment with difference number of blocks
    def switch_match_diff_env(self, obj_id, goal_num_block, replace=False):
        # get new obs_transit
        new_obs_transit = copy.deepcopy(self.obs_transit)
        new_obs_transit.set_num(goal_num_block)

        new_rules = []
        for self_rule in self.rules:
            new_rules.append([])
            for term in self_rule:
                new_term = self._switch_helper(term, obj_id, new_obs_transit)
                new_rules[-1].append(new_term)
        
        new_lagrangian_rules = []
        if self.lagrangian_rules is not None:
            for self_rule in self.lagrangian_rules:
                new_lagrangian_rules.append([])
                for term in self_rule:
                    new_term = self._switch_helper(term, obj_id, new_obs_transit)
                    new_lagrangian_rules[-1].append(new_term)

        new_non_lagrangian_rules = []
        if self.non_lagrangian_rules is not None:
            for self_rule in self.non_lagrangian_rules:
                new_non_lagrangian_rules.append([])
                for term in self_rule:
                    new_term = self._switch_helper(term, obj_id, new_obs_transit)
                    new_non_lagrangian_rules[-1].append(new_term)

        if replace:
            self.set_new_rules(new_rules, new_lagrangian_rules, new_non_lagrangian_rules)
            self.obs_transit = new_obs_transit

        return new_rules

    def get_obj_id(self):
        contain_objs = []
        for self_rule in self.rules:
            for term in self_rule:
                ori_id = int(term[term.find('[')+1 : term.find(']')])
                contain_objs.append(self.obs_transit.get_obj_id(ori_id))
        
        return contain_objs

    def get_obs_id(self):
        contain_obs = []
        for self_rule in self.rules:
            for term in self_rule:
                ori_id = int(term[term.find('[')+1 : term.find(']')])
                contain_obs.append(ori_id)
        
        return contain_obs

    def get_reward_values(self):
        contain_vals = []
        for self_rule in self.rules:
            for term in self_rule:
                if '>' in term:
                    val = float(term[term.find('>')+1:])
                else:
                    val = float(term[term.find('<=')+2:])
                contain_vals.append(val)
        
        return contain_vals

    def execute(self, obs):
        if self.obs_transit is not None:
            obs = self.obs_transit.get_abs_obs(obs)
        obs = np.squeeze(obs, axis=0)
        reward = -eval(self.comb_dense_rule)
        if self.lagrangian_rules is None:
            cost = 0
        else:
            cost = eval(self.lagrangian_comb_dense_rule)
        if self.non_lagrangian_rules is None:
            cost = cost
        else:
            cost = max(cost, 0.1-eval(self.non_lagrangian_comb_dense_rule))

        return reward, cost

    def execute_rules(self, obs):
        if self.obs_transit is not None:
            obs = self.obs_transit.get_abs_obs(obs)
        obs = np.squeeze(obs, axis=0)
        if self.lagrangian_rules is None:
            try:
                bool_res = eval(self.comb_rule)
            except:
                pdb.set_trace()
        else:
            bool_res = eval(self.comb_rule) and eval(self.lagrangian_comb_rule)
        if self.non_lagrangian_rules is not None:
            bool_res = bool_res and not eval(self.non_lagrangian_comb_rule)
        return bool_res

    def execute_details(self, obs):
        if self.obs_transit is not None:
            obs = self.obs_transit.get_abs_obs(obs)
        obs = np.squeeze(obs, axis=0)

        results = []
        for rule in self.dense_rules:
            results.append(-eval(rule))

        if self.lagrangian_rules is not None:
            for rule in self.lagrangian_dense_rules:
                results.append(-eval(rule))

        return results

    def get_copy(self):
        new_equ = TreeContDenseLagrangianCls(self.rules, self.eps, copy.deepcopy(self.obs_transit))
        if self.lagrangian_rules is not None:
            new_equ.comb_lagrangian(self.lagrangian_rules)
        if self.non_lagrangian_rules is not None:
            new_equ.comb_non_lagrangian(self.non_lagrangian_rules)
        return new_equ

    def split_predicate(self, samples=None):
        # extract rules
        all_cls = [TreeContDenseLagrangianCls([rule], self.eps, self.obs_transit) for rule in self.rules]
        
        # sort rules based on success rate on samples
        success_rates = [0 for _ in all_cls]
        for sample in samples:
            for cls_id, each_cls in enumerate(all_cls):
                success_rates[cls_id] += int(each_cls.execute_rules(np.expand_dims(sample, axis=0)))

        all_cls = [each_cls for _, each_cls in sorted(zip(success_rates, all_cls), key=lambda x: x[0], reverse=True)]

        return all_cls

    def __str__(self):
        print_rule = ' or\n'.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.rules])
        print_lagrangian_rule = ''
        if self.lagrangian_rules is not None:
            print_lagrangian_rule = '\n const:\n' + ' or\n'.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.lagrangian_rules])
        print_non_lagrangian_rule = ''
        if self.non_lagrangian_rules is not None:
            print_non_lagrangian_rule = '\n non const:\n' + 'not ' + ' or\n'.join(['( ' + ' and '.join(rule)  + ' )' for rule in self.non_lagrangian_rules])
        print_dense_rule = ' *\n'.join(self.dense_rules)
        print_lagrangian_dense_rule = ''
        if self.lagrangian_rules is not None:
            print_lagrangian_dense_rule = '\n const:\n' + ' *\n'.join(self.lagrangian_dense_rules)
        print_non_lagrangian_dense_rule = ''
        if self.non_lagrangian_rules is not None:
            print_non_lagrangian_dense_rule = '\n non const:\n' + '- ' + ' *\n'.join(self.non_lagrangian_dense_rules)

        return print_rule + print_lagrangian_rule + print_non_lagrangian_rule + \
              '\n\n' + print_dense_rule + print_lagrangian_dense_rule + print_non_lagrangian_dense_rule


class TreeCls:
    def __init__(self, tree_type='tree', obs_transit=None, dense_type='cut', only_old=False, lagrangian=False, simplify=False, **kwargs):
        self.tree_type = tree_type
        if tree_type == 'tree':
            # self.tree = tree.DecisionTreeClassifier(min_samples_leaf=25)
            # self.tree = tree.DecisionTreeClassifier(min_samples_leaf=15)
            self.tree = tree.DecisionTreeClassifier(**kwargs)
        elif tree_type == 'forest':
            self.tree = RandomForestClassifier()

        self.dense_type = dense_type
        self.lagrangian = lagrangian
        # for debug
        self.obs_transit = obs_transit
        self.only_old = only_old
        self.simplify = simplify

    def do_fit(self, obs, label):
        if self.obs_transit is not None:
            obs = self.obs_transit.get_abs_obs(obs)
        self.tree.fit(obs, label)

    def execute(self, obs):
        if self.obs_transit is not None:
            obs = self.obs_transit.get_abs_obs(obs)
        output = self.tree.predict(obs).item()

        return output
    
    def extract_dense(self):
        tree_ = self.tree.tree_
        feature_name = [
            'obs[{}]'.format(i) if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        path = []
        paths = {0:[], 1:[]}

        def special_case(node):
            left_child = tree_.children_left[node]
            right_child = tree_.children_right[node]

            # left
            if tree_.feature[left_child] != _tree.TREE_UNDEFINED:
                left_special = special_case(left_child)
            else:
                left_special = tree_.value[left_child][0,1] > 0

            # right
            if tree_.feature[right_child] != _tree.TREE_UNDEFINED:
                right_special = special_case(right_child)
            else:
                right_special = tree_.value[right_child][0,1] > 0

            return left_special and right_special

        def recurse(node, path, paths):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                # check special case: if left and right both contain positive node
                if node != 0 and self.simplify and special_case(node):
                    paths[1].append(p1)
                else:
                    p1 += [f"{name}<={threshold}"]
                    recurse(tree_.children_left[node], p1, paths)
                    p2 += [f"{name}>{threshold}"]
                    recurse(tree_.children_right[node], p2, paths)
            else:
                if tree_.value[node][0,0] > 0:
                    paths[0].append(path)
                if tree_.value[node][0,1] > 0:
                # else:
                    paths[1].append(path)

        recurse(0, path, paths)

        if self.dense_type == 'cut':
            return TreeCutDenseCls(paths[1], obs_transit=self.obs_transit)
        elif self.dense_type == 'cont':
            if self.lagrangian:
                return TreeContDenseLagrangianCls(paths[1], obs_transit=self.obs_transit, only_old=self.only_old)
            else:
                return TreeContDenseCls(paths[1], obs_transit=self.obs_transit, only_old=self.only_old)


    def print_tree(self, path):
        tree.plot_tree(self.tree)
        plt.savefig(path)
        plt.close()

class TreeLearner:
    def __init__(self, tree_type, obs_transit=None, dense_type='cut', only_old=False, lagrangian=False, simplify=False, **kwargs):
        self.tree_type = tree_type
        self.dense_type = dense_type
        self.kwargs = kwargs

        # for obs trainsit
        self.obs_transit = obs_transit
        self.only_old = only_old
        self.lagrangian = lagrangian
        self.simplify = simplify

    def reset(self):
        self.treeCls = TreeCls(self.tree_type, obs_transit=self.obs_transit, dense_type=self.dense_type, only_old=self.only_old, \
                               lagrangian=self.lagrangian, simplify=self.simplify, **self.kwargs)

    def do_learn(self, X, y):
        self.treeCls.do_fit(X, y)

        return self.treeCls
    
    def print_tree(self, path):
        self.treeCls.print_tree(path)