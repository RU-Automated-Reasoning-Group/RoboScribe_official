import numpy as np
import copy
import torch

import pdb

class Reward:
    def __init__(self):
        pass

    def get_reward(self, obs, env_rew):
        raise NotImplementedError
    
    def set_train(self):
        pass

    def set_eval(self):
        pass

    def get_copy(self):
        return copy.deepcopy(self)

    def adjust_equ(self, data):
        pass

    def comb_rules(self, new_rules, comb_method='and'):
        pass

    def split_rew(self):
        return self

    def __str__(self):
        return ''

class RewIdentity(Reward):
    def __init__(self):
        super().__init__()
        self.train = True

    def get_reward(self, obs, env_rew):
        return env_rew
    
    def __str__(self):
        return '<Environment Reward>'
    
class RewTBD(Reward):
    def __init__(self):
        super().__init__()
        self.train = True

    def get_reward(self, obs, env_rew):
        return 0
    
    def __str__(self):
        return '<TBD Reward>'

class RewDSO(Reward):
    def __init__(self, equ, train=True):
        super().__init__()
        self.equ = equ
        self.train = train

    def get_reward(self, obs, env_rew):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if self.train:
            return self.equ.execute(obs)
        else:
            return self.equ.execute_rules(obs)

    def set_train(self):
        self.train = True

    def set_eval(self):
        self.train = False

    def get_copy(self):
        return RewDSO(self.equ.get_copy(), self.train)

    def adjust_equ(self, data):
        self.equ.adjust_rules(data)

    def comb_rules(self, new_rules, comb_method='and'):
        return self.equ.comb_rules(new_rules, comb_method)

    def split_rew(self, samples):
        return [RewDSO(equ, self.train) for equ in self.equ.split_predicate(samples)]

    def __str__(self):
        try:
            return str(self.equ.pretty())
        except:
            # return 'special reward'
            return str(self.equ)

class RewCritic(Reward):
    def __init__(self, policy, train=True):
        super().__init__()
        self.policy = policy
        self.train = train

    def get_reward(self, obs, env_rew):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            act, _ = self.policy.select_action(obs, False)
            q_vals = [self.policy.critic_1(obs, act).item(),
                      self.policy.critic_2(obs, act).item()]
            rew = min(q_vals)
            # act, _ = self.policy.predict(obs, deterministic=True)
            # q_vals = torch.cat(self.policy.critic(torch.tensor(obs).unsqueeze(0), 
            #                                       torch.tensor(act).unsqueeze(0)), dim=1)
            # rew, _ = torch.min(q_vals, dim=1, keepdim=True)
            # rew = rew.item()

        return rew

    def set_train(self):
        self.train = True

    def set_eval(self):
        self.train = False

    def get_copy(self):
        # return RewCritic(self.policy.do_copy(), self.train)
        return RewCritic(self.policy, self.train)

    def __str__(self):
        return 'critic rew'

class RewDebug(Reward):
    def __init__(self, equ):
        super().__init__()
        self.equ = equ

    def get_reward(self, obs, env_rew):
        return self.equ(obs, env_rew)
    
    def __str__(self):
        return "special reawrd"