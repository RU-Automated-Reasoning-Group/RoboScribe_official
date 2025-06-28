

class Task:
    def __init__(self, policy, reward_fun):
        self.policy = policy
        self.reward_fun = reward_fun

    def train_policy(self):
        pass

    def rollout(self):
        pass