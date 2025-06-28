import torch
import numpy as np
from policy.rce.replaybuffer import UniformReplayBuffer, N_step_traj
from policy.rce.models import tanh_gaussian_actor, Critic
from torch.distributions.normal import Normal
import copy
import os
from tqdm import tqdm

import pdb

import time

class RceReward:
    def __init__(self, actor, critic_1, critic_2, max_action, new_device='cpu'):
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.max_action = max_action

        self.device = new_device
        self.actor = self.actor.to(new_device)
        self.critic_1 = self.critic_1.to(new_device)
        self.critic_2 = self.critic_2.to(new_device)

    def select_action(self, s, rsample=False, return_action_log_probs=False):

        mean, std = self.actor(s)
        pre_tanh_action = Normal(mean, std).sample()
        action = torch.tanh(pre_tanh_action).detach()

        if return_action_log_probs:
            actions_probs = Normal(mean, std).log_prob(pre_tanh_action) - torch.log(1 - action ** 2 + 1e-6)
            actions_probs = actions_probs.sum(dim=1, keepdim=True)
            return action, pre_tanh_action, actions_probs

        return action, pre_tanh_action
    
    def predict(self, s, deterministic=True):
        action, pre_tanh_action = self.select_action(self.to_tensor(s[None, :]), not deterministic)
        action = action.cpu().numpy().squeeze(0)
        return action * self.max_action, pre_tanh_action

    def to_tensor(self, x, type=torch.float32):
        return torch.tensor(x, dtype=type).to(self.device)

    def do_copy(self):
        return RceReward(copy.deepcopy(self.actor), copy.deepcopy(self.critic_1), copy.deepcopy(self.critic_2), self.max_action, self.device)

    def save(self, path):
        torch.save({
            'actor_model': self.actor.state_dict(),
            'q1_model': self.critic_1.state_dict(),
            'q2_model': self.critic_2.state_dict()
        }, path)

class RceAgent:
    def __init__(self, env, eval_env, args, env_params, expert_examples_buffer, logger, device='cuda'):
        self.env = env
        self.eval_env = eval_env
        self.args = args
        self.env_params = env_params
        self.device = device
        self.max_action = self.env_params['max_action']
        self.n_steps_traj_rec = N_step_traj(n_steps=self.args.n_steps)
        self.reply_buffer = UniformReplayBuffer(max_size=self.args.buffer_size)
        self.expert_buffer = expert_examples_buffer
        self.critic_criterion = torch.nn.MSELoss(reduction='none')
        self.logger = logger

        # entropy regularizer only for SAC, for RCE the entropy coef is fixed
        self.target_entropy = -np.prod(self.env.action_space.shape).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.actor_lr)

        self.actor = tanh_gaussian_actor(self.env_params['state_dim'], self.env_params['action_dim'],
                                         self.args.hidden_size,
                                         self.args.log_std_min, self.args.log_std_max).to(self.device)
        self.critic_1 = Critic(self.env_params['state_dim'], self.env_params['action_dim'], self.args.hidden_size,
                               loss_type='c').to(self.device)
        self.critic_2 = Critic(self.env_params['state_dim'], self.env_params['action_dim'], self.args.hidden_size,
                               loss_type='c').to(self.device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)

        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_1_optim = torch.optim.Adam(self.critic_1.parameters(),
                                               lr=self.args.critic_lr)
        self.critic_2_optim = torch.optim.Adam(self.critic_2.parameters(),
                                               lr=self.args.critic_lr)

        self.global_step = 0

    def do_copy(self):
        new_agent = RceAgent(self.env, self.eval_env, self.args, self.env_params, self.expert_buffer)
        new_agent.actor.load_state_dict(self.actor.state_dict())
        new_agent.critic_1.load_state_dict(self.critic_1.state_dict())
        new_agent.critic_2.load_state_dict(self.critic_2.state_dict())
        new_agent.target_critic_1 = copy.deepcopy(new_agent.critic_1).to(new_agent.device)
        new_agent.target_critic_2 = copy.deepcopy(new_agent.critic_2).to(new_agent.device)

        return new_agent


    def update(self):
        # critic loss is "c" -> rce
        expert_transitions = self.expert_buffer.sample(self.args.batch_size)
        expert_states = self.to_tensor(expert_transitions)  # success examples

        transitions = self.reply_buffer.sample(self.args.batch_size)
        processed_batches = np.asarray([self._preprocess_n_steps_traj_batches(x) for x in transitions])

        # s_{t},a_{t},s_{t+1},s_{t+n},d_{t+n}
        states, actions, next_states, future_states, _ = map(np.array, zip(*processed_batches))
        states = self.to_tensor(states)
        actions = self.to_tensor(actions)
        next_states = self.to_tensor(next_states)
        future_states = self.to_tensor(future_states)

        # compute the targets
        with torch.no_grad():
            next_actions, _ = self.select_action(next_states)
            target_q_1 = self.target_critic_1(next_states, next_actions)
            target_q_2 = self.target_critic_2(next_states, next_actions)

            future_actions, _ = self.select_action(future_states)
            target_q_future_1 = self.target_critic_1(future_states, future_actions)
            target_q_future_2 = self.target_critic_2(future_states, future_actions)

            gamma_n = self.args.gamma ** self.args.n_steps
            target_q_1 = (target_q_1 + gamma_n * target_q_future_1) / 2.0
            target_q_2 = (target_q_2 + gamma_n * target_q_future_2) / 2.0

            target_q = torch.min(target_q_1, target_q_2)

            w = target_q / (1. - target_q)
            td_targets = self.args.gamma * w / (1. + self.args.gamma * w)

        td_targets = torch.cat([torch.ones(self.args.batch_size, 1).to(self.device), td_targets], dim=0)
        weights = torch.cat(
            [torch.ones(self.args.batch_size, 1).to(self.device) - self.args.gamma, 1. + self.args.gamma * w],
            dim=0)

        # compute the predictions
        expert_actions, _ = self.select_action(expert_states)
        pred_expert_1 = self.critic_1(expert_states, expert_actions)
        pred_expert_2 = self.critic_2(expert_states, expert_actions)

        pred_1 = self.critic_1(states, actions)
        pred_2 = self.critic_2(states, actions)

        pred_1 = torch.cat([pred_expert_1, pred_1], dim=0)
        pred_2 = torch.cat([pred_expert_2, pred_2], dim=0)

        critic_1_loss = (weights * self.critic_criterion(pred_1, td_targets)).mean()
        critic_2_loss = (weights * self.critic_criterion(pred_2, td_targets)).mean()

        self.critic_1_optim.zero_grad()
        critic_1_loss.backward(retain_graph=True)
        self.critic_1_optim.step()
        self.critic_2_optim.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optim.step()

        # soft update the target critic networks
        self._soft_update_target_network(self.target_critic_1, self.critic_1)
        self._soft_update_target_network(self.target_critic_2, self.critic_2)

        # update the actor
        actions_new, _, log_probs = self.select_action(states, return_action_log_probs=True)
        target_q1 = self.critic_1(states, actions_new)
        target_q2 = self.critic_2(states, actions_new)
        targets_q = torch.min(target_q1, target_q2)
        actor_loss = (self.args.entropy_coef * log_probs - targets_q).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return critic_1_loss.item(), critic_2_loss.item(), actor_loss.item()

    def collect_rollouts(self):
        num_n_step_traj = 0
        s, _ = self.env.reset()
        for i in range(self.args.rollout_steps):
            action, _ = self.select_action(self.to_tensor(s[None, :]), False)
            action = action.cpu().numpy().squeeze(0)
            s_, r, done, truncated, _ = self.env.step(action * self.max_action)
            done = done or truncated
            if_complete = self.n_steps_traj_rec.add(s, action, r, s_, done)

            if if_complete:
                if not done:
                    self.reply_buffer.add_n_step_traj(self.n_steps_traj_rec.dump())
                    num_n_step_traj += 1
                self.n_steps_traj_rec.reset()
            s = s_
            if done:
                s, _ = self.env.reset()
        return num_n_step_traj

    def _evaluate(self):
        episode_rew = 0
        for i in range(self.args.evaluation_rollouts):
            s, _ = self.eval_env.reset()
            while True:
                action, _ = self.select_action(self.to_tensor(s[None, :]), False)
                action = action.cpu().numpy().squeeze(0)
                s_, r, done, truncated, _ = self.eval_env.step(action * self.max_action)
                done = done or truncated
                episode_rew += r
                s = s_
                if done:
                    break
        return episode_rew / float(self.args.evaluation_rollouts)

    def learn(self):
        self.init_replay_buffer()
        for i in tqdm(range(self.args.iterations)):
            pdb.set_trace()
            num_n_step_traj = self.collect_rollouts()
            for _ in range(num_n_step_traj):
                q1_loss, q2_loss, actor_loss = self.update()

                self.global_step += 1
                # if self.global_step % self.args.save_model_interval == 0:
                #     eval_rew = self._evaluate()
                #     print('episode return:{}, global_step:{}'.format(eval_rew, self.global_step))
                #     # self.writer.add_scalar('episode return', eval_rew, self.global_step)
                #     self.logger.add_scalar('Eval/episode_return', eval_rew, self.global_step)

                self.logger.add_scalar('Train/q1_loss', q1_loss, self.global_step)
                self.logger.add_scalar('Train/q2_loss', q2_loss, self.global_step)
                self.logger.add_scalar('Train/actor_loss', actor_loss, self.global_step)

    def _preprocess_n_steps_traj_batches(self, x):
        """
        output computed variables for "SAC" and "RCE"
        :param x: single sample
        :return: "SAC":(s,a,r,s_future,done)
                "RCE": s,a,s_next,s_future,done
        """
        states, actions, rewards, dones, final_state = x
        state, next_state, future_state = states[0], states[1], final_state
        action = actions[0]
        done = dones[-1]
        return state, action, next_state, future_state, done

    def select_action(self, s, rsample=True, return_action_log_probs=False):

        mean, std = self.actor(s)
        if rsample:
            pre_tanh_action = mean + torch.randn(mean.size()).to(self.device) * std
            action = torch.tanh(pre_tanh_action)
            action.requires_grad_()
        else:
            pre_tanh_action = Normal(mean, std).sample()
            action = torch.tanh(pre_tanh_action).detach()

        if return_action_log_probs:
            actions_probs = Normal(mean, std).log_prob(pre_tanh_action) - torch.log(1 - action ** 2 + 1e-6)
            actions_probs = actions_probs.sum(dim=1, keepdim=True)
            return action, pre_tanh_action, actions_probs

        return action, pre_tanh_action

    def predict(self, s, deterministic=True):
        return self.select_action(s, not deterministic)

    def init_replay_buffer(self):
        init_collected_n_steps_traj = 0
        while init_collected_n_steps_traj < self.args.init_num_n_step_trajs:
            num = self.collect_rollouts()
            init_collected_n_steps_traj += num

    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def to_tensor(self, x, type=torch.float32):
        return torch.tensor(x, dtype=type).to(self.device)

    def save(self, path):
        torch.save({
            'actor_model': self.actor.state_dict(),
            'q1_model': self.critic_1.state_dict(),
            'q2_model': self.critic_2.state_dict()
        }, path)