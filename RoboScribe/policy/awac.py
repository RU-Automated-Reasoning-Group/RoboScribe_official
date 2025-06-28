import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional
import wandb
from tqdm import trange, tqdm

TensorBatch = List[torch.Tensor]

ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")

import cv2
import pdb

IMG_IDX = 0

@dataclass
class TrainConfig:
    project: str = "Abs-Demo"
    group: str = "AWAC"
    name: str = "AWAC"
    checkpoints_path: Optional[str] = None

    # env_name: str = "customBlock"
    # env_name: str = "tower4_2"
    # env_name: str="tower4_1.5e7_seed2000"
    # env_name: str="tower5_1.5e7_seed4000"
    # env_name: str="tower5_away_1.5e7_seed1000"
    # env_name: str="pickplacecube_3_crop_1.5e7_seed0_cont_2"
    # env_name: str="pickplacecube_3_crop_1.5e7_seed0_2"
    # env_name: str="drawer_pickplacecube_reassign_3_crop_1.5e7_seed0_4"
    # env_name: str="tower5_away_1.5e7_seed0_continue_2"
    # env_name: str="tower5_away_1.5e7_seed0_non_terminal"
    # env_name: str = 'pickplace4_50000_newparam'
    env_name: str = 'pickplace2'
    # env_name: str = 'drawer_cube_1.5e7'
    # 0, 1000, 3000, 4000
    # env_name: str = 'push_3_1.5e7_seed2000'
    # env_name: str = 'push_3_deepset_2e7'
    # env_name: str = 'push_block1_1'
    # env_name: str = 'multitower4_2'
    # env_name: str = 'multitower5_1'
    seed: int = 0
    eval_seed: int = 0  # Eval environment seed
    test_seed: int = 2000
    deterministic_torch: bool = True
    device: str = "cuda"
    # device: str = "cpu"

    buffer_size: int = 20_000_000
    offline_iterations: int = int(20000)  # Number of offline updates
    # offline_iterations: int = int(50000)  # Number of offline updates
    # offline_iterations: int = int(100000)  # Number of offline updates
    # offline_iterations: int = int(200000)  # Number of offline updates
    # offline_iterations: int = int(1000000)  # Number of offline updates
    # offline_iterations: int = int(0)  # Number of offline updates

    # online_iterations: int = int(40000)  # Number of online updates
    # online_iterations: int = int(100000)  # Number of online updates
    # online_iterations: int = int(2000000)  # Number of online updates
    # online_iterations: int = int(10000000)  # Number of online updates
    online_iterations: int = int(15000000)  # Number of online updates
    # online_iterations: int = int(20000000)  # Number of online updates
    batch_size: int = 256
    # eval_frequency: int = 3000
    # eval_frequency: int = 5000
    # eval_frequency: int = 10000
    # eval_frequency: int = 20000
    eval_frequency: int = 40000
    # n_test_episodes: int = 6
    n_test_episodes: int = 100
    normalize_reward: bool = False

    # group 1
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    awac_lambda: float = 1.0

    # group 2
    # hidden_dim: int = 256
    # learning_rate: float = 0.001
    # gamma: float = 0.95
    # tau: float = 0.05
    # awac_lambda: float = 1.0

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)


def set_env_seed(env: Optional[gym.Env], seed: int):
    env.seed(seed)
    env.action_space.seed(seed)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        min_action: float = -3.0,
        max_action: float = 3.0,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        self._min_action = min_action
        self._max_action = max_action

    def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(self._min_action, self._max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self._mlp.training:
            action_t = policy.sample()
        else:
            action_t = policy.mean
        action = action_t[0].cpu().numpy()
        return action

class DeepsetActor(Actor):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        min_action: float = -3.0,
        max_action: float = 3.0,
        agg: str='mean',
        obs_transit=None
    ):
        super().__init__(state_dim, action_dim, hidden_dim, min_log_std, max_log_std, min_action, max_action)
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self._act_out = nn.Linear(hidden_dim, action_dim)
        self.agg = agg
        self.obs_transit = obs_transit

    def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        # preprocess state
        if self.obs_transit is not None:
            state_dict = self.obs_transit.do_crop_state_collect(state, add_on=True)
            objs = torch.concat([block.unsqueeze(1) for block in state_dict['block']], dim=1)
            grip = state_dict['gripper'].unsqueeze(1).expand(-1, objs.shape[1], -1)
            goals = torch.concat([goal.unsqueeze(1) for goal in state_dict['goal']], dim=1)
            add_on = state_dict['add_on'].unsqueeze(2)
            state = torch.concat([grip, objs, goals, add_on], dim=-1)

        fea = self._mlp(state)
        if self.agg == 'sum':
            fea = fea.sum(-2)
        elif self.agg == 'mean':
            fea = fea.mean(-2)
        elif self.agg == 'max':
            fea = fea.max(-2).values
        else:
            raise ValueError(f'Unrecognized aggregation function {self.agg}')
        mean = self._act_out(fea)

        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy



class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value = self._mlp(torch.cat([state, action], dim=-1))
        return q_value

class DeepsetCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        obs_transit=None
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.obs_transit = obs_transit

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # preprocess state
        if self.obs_transit is not None:
            state_dict = self.obs_transit.do_crop_state_collect(state, add_on=True)
            objs = torch.concat(state_dict['block'], dim=1)
            grip = state_dict['gripper'].clone()
            goals = torch.concat(state_dict['goal'], dim=1)
            add_on = state_dict['add_on'].clone()
            state = torch.concat([grip, objs, goals, add_on], dim=-1)

        q_value = self._mlp(torch.cat([state, action], dim=-1))
        return q_value



def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class AdvantageWeightedActorCritic:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 5e-3,  # parameter for the soft target update,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
    ):
        self._actor = actor
        self._actor_optimizer = actor_optimizer

        self._critic_1 = critic_1
        self._critic_1_optimizer = critic_1_optimizer
        self._target_critic_1 = deepcopy(critic_1)

        self._critic_2 = critic_2
        self._critic_2_optimizer = critic_2_optimizer
        self._target_critic_2 = deepcopy(critic_2)

        self._gamma = gamma
        self._tau = tau
        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

    def _actor_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            pi_action, _ = self._actor(states)
            v = torch.min(
                self._critic_1(states, pi_action), self._critic_2(states, pi_action)
            )

            q = torch.min(
                self._critic_1(states, actions), self._critic_2(states, actions)
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self._awac_lambda), self._exp_adv_max
            )

        action_log_prob = self._actor.log_prob(states, actions)
        loss = (-action_log_prob * weights).mean()
        return loss

    def _critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_actions, _ = self._actor(next_states)

            q_next = torch.min(
                self._target_critic_1(next_states, next_actions),
                self._target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self._gamma * (1.0 - dones) * q_next

        q1 = self._critic_1(states, actions)
        q2 = self._critic_2(states, actions)

        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = q1_loss + q2_loss
        return loss

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
    ):
        loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        return loss.item()

    def _update_actor(self, states, actions):
        loss = self._actor_loss(states, actions)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()

    def predict(self, obs, deterministic=True):
        return self._actor.act(obs, "cuda"), None

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        critic_loss = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss = self._update_actor(states, actions)

        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)

        result = {"critic_loss": critic_loss, "actor_loss": actor_loss}
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self._actor.state_dict(),
            "critic_1": self._critic_1.state_dict(),
            "critic_2": self._critic_2.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._actor.load_state_dict(state_dict["actor"])
        self._critic_1.load_state_dict(state_dict["critic_1"])
        self._critic_2.load_state_dict(state_dict["critic_2"])

    def save(self, param_path):
        state_dict = self.state_dict()
        torch.save(state_dict, param_path)


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        set_env_seed(env, seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env


def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward


@torch.no_grad()
def eval_actor(
    dataset, env: gym.Env, obs_transit, all_rew_fun, add_rew_fun, state_mean, state_std, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []

    # pdb.set_trace()
    if not os.path.exists("debug_imgs/eval"):
        os.mkdir("debug_imgs/eval")
    
    success_count = 0
    success_goal_idx = set()
    # pdb.set_trace()
    for _ in range(n_episodes):
        state, done = env.reset()[0], False
        episode_reward = 0.0
        episode_len = 0

        crop_id = 1
        while (not done) and (episode_len <= 100):
            input_state = obs_transit.do_crop_state(np.array([state]), crop_id)
            input_state = (input_state - state_mean) / state_std
            action = actor.act(input_state, device) # / 2.0

            # calculate most similar state
            # min_idx = np.argmin(np.sum((dataset["observations"] - input_state) ** 2, axis=1))
            # print("current observation", input_state)
            # print("closed dataset observation", dataset["observations"][min_idx])
            # print("current action", action)
            # print("closest dataset action", dataset["actions"][min_idx])
            # groundtruth_action = dataset["actions"][min_idx]
            # new_observations.append(input_state.copy())
            # new_actions.append(groundtruth_action.copy())
            # import pdb
            # pdb.set_trace()
            state, reward, terminate, truncate, _ = env.step(action[0])
            rwd, cost = all_rew_fun[crop_id - 1].get_reward(state, -1)
            rwd = add_rew_fun(rwd, cost, all_rew_fun[crop_id - 1], False)
            # print(rwd)
            done = terminate or truncate
            if rwd > 0.9:
                print("one block succeed")
                if crop_id == 2:
                    done = True
                    success_count += 1
                    success_goal_idx.add(env.env.env._env.goal_idx)
                crop_id = 2
            episode_reward += rwd
            episode_len += 1
            img = env.render()
            global IMG_IDX
            cv2.imwrite(f"debug_imgs/eval/{IMG_IDX}.png", img)
            IMG_IDX += 1
        episode_rewards.append(episode_reward)

    IMG_IDX = 0

    actor.train()
    print(f"success episodes: {success_count}")
    print(f"success goal idx {success_goal_idx}")
    return np.asarray(episode_rewards)

# @torch.no_grad()
# def eval_actor(
#     env: gym.Env, actor: Actor, device: str, n_episodes: int, seed: int
# ) -> Tuple[np.ndarray, np.ndarray]:
#     env.seed(seed)
#     actor.eval()
#     episode_rewards = []
#     successes = []
#     for _ in range(n_episodes):
#         state, done = env.reset(), False
#         episode_reward = 0.0
#         goal_achieved = False
#         while not done:
#             action = actor.act(state, device)
#             state, reward, done, env_infos = env.step(action)
#             episode_reward += reward
#             if not goal_achieved:
#                 goal_achieved = is_goal_reached(reward, env_infos)
#         # Valid only for environments with goal
#         successes.append(float(goal_achieved))
#         episode_rewards.append(episode_reward)

#     actor.train()
#     return np.asarray(episode_rewards), np.mean(successes)


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset: Dict, env_name: str, max_episode_steps: int = 1000) -> Dict:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        return {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return {}


def modify_reward_online(reward: float, env_name: str, **kwargs) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    elif "antmaze" in env_name:
        reward -= 1.0
    return reward


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

@torch.no_grad()
def eval_actor_scratch(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int, traj_limit: int
) -> np.ndarray:
    if traj_limit is None:
        traj_limit = 150

    env.seed(seed)
    env.set_eval()
    actor.eval()

    episode_rewards = []

    # pdb.set_trace()
    if not os.path.exists("debug_imgs/eval"):
        os.mkdir("debug_imgs/eval")
    
    # pdb.set_trace()
    success_id = set()
    success_num = 0
    for _ in trange(n_episodes):
        state, done = env.reset()[0], False
        episode_reward = 0.0
        episode_len = 0

        reset_len = env.traj_len - env.traj_id
        while (not done) and (episode_len <= min(traj_limit, reset_len)):
            action = actor.act(state, device) # / 2.0

            state, reward, terminate, truncate, info = env.step(action)

            # debug: test
            # print(rwd)
            # if 'stage_done' in info and info['stage_done']:
            #     print('stage success')
            
            done = terminate or truncate
            
            if done and 'stage_done' in info and info['stage_done']:
                # print('episode success')
                success_num += 1
                # success_id.add(env.env.env.env._env.goal_idx)

            episode_reward += reward
            episode_len += 1
            img = env.render()
            global IMG_IDX
            # cv2.imwrite(f"debug_imgs/eval/{IMG_IDX}.png", img)
            IMG_IDX += 1
        episode_rewards.append(episode_reward)

    IMG_IDX = 0

    print('success num: {}'.format(success_num))
    # print('success id: {}'.format(success_id))

    actor.train()
    return np.asarray(episode_rewards)

@torch.no_grad()
def eval_actor_multi(
    env: gym.Env, actor_seq: List[nn.Module], device: str, n_episodes: int, seed: int, traj_limit: int, node_policy_dict: Dict, store_path=None, \
    final_rew_fun=None, loop_limit=None, train_log=False, extra_reset_fun=False, debug_reward_fun=None,
) -> np.ndarray:
    debug_store = True

    if traj_limit is None:
        traj_limit = 150

    env.seed(seed)
    env.set_eval()
    for actor in actor_seq:
        actor.eval()

    episode_rewards = []

    # pdb.set_trace()
    # if store_path is None:
    #     store_path = "debug_imgs/eval"
    if store_path is not None:
        if not os.path.exists(store_path):
            os.mkdir(store_path)
    
    # pdb.set_trace()
    success_id = set()
    success_num = 0
    test_case = []
    if train_log:
        success_imgs = []
        fail_imgs = []
    for episode_id in trange(n_episodes):
        while True:
            try:
                state, done = env.reset()[0], False
                break
            except:
                pass

        episode_reward = 0.0
        episode_len = 0
        # debug
        test_case.append(0)

        # only for debug
        debug_states = []

        if debug_reward_fun is not None:
            debug_check = False
        reset_len = env.traj_len - env.traj_id
        loop_count = 0
        if train_log:
            img_list = []

        # only for debug
        # assert node_policy_dict[env.cur_id] == 4

        # pdb.set_trace()

        while (not done) and (episode_len <= min(traj_limit, reset_len)):
            if loop_limit is not None and loop_count > loop_limit:
                reset_success = env.loop_reset(env.last_raw_obs)
                # special success
                if not reset_success:
                    test_case[-1] = -1
                    success_num += 1
                    break

                if extra_reset_fun:
                    # extra_reset_fun()
                    # debug only for pickplacecube3
                    env.env.reassign_cubeA()
                # pdb.set_trace()
                loop_count = 0

            actor = actor_seq[node_policy_dict[env.cur_id]]
            action = actor.act(state, device) # / 2.0

            state, reward, terminate, truncate, info = env.step(action)

            # only for debug
            # debug_states.append(np.expand_dims(env.last_raw_obs, 0))
            # debug_states.append(env.cur_id)

            # check if valid for debug
            if debug_reward_fun is not None:
                raw_obs = env.last_raw_obs
                if debug_reward_fun.get_reward(np.expand_dims(raw_obs, 0), 0):
                    debug_check = True

            # debug: test
            # print(rwd)
            if 'stage_done' in info and info['stage_done']:
                test_case[-1] += 1
                loop_count = 0
                # print('stage success')
            
            done = terminate or truncate
            
            if terminate and 'stage_done' in info and info['stage_done']:
                # final reward function (only for debug)
                if (not info['env_success']) and final_rew_fun is not None:
                    # pdb.set_trace()
                    raw_obs = env.last_raw_obs
                    if final_rew_fun.get_reward(np.expand_dims(raw_obs, 0), 0):
                        test_case[-1] = -1
                        success_num += 1
                else:
                    test_case[-1] = -1
                    # print('episode success')
                    success_num += 1
                    # success_id.add(env.env.env.env._env.goal_idx)
                # test_case[-1] = 100
                # print('episode success')
                # success_num += 1

            episode_reward += reward
            episode_len += 1
            loop_count += 1
            img = env.render()
            if train_log:
                img_list.append(np.transpose(img, (2,0,1)))
            global IMG_IDX
            if store_path is not None:
                if not os.path.exists(f"{store_path}/{episode_id}"):
                    os.makedirs(f"{store_path}/{episode_id}")
                cv2.imwrite(f"{store_path}/{episode_id}/{episode_len}.png", img)
                # cv2.imwrite(f"{store_path}/{episode_id}/{node_policy_dict[env.cur_id]}_{episode_len}.png", img)
            IMG_IDX += 1

        # print(IMG_IDX)
        # pdb.set_trace()
            
        # debug_states = np.concatenate(debug_states, axis=0)

        episode_rewards.append(episode_reward)
        if debug_reward_fun is not None and debug_check:
            test_case[-1] = 'special'

        if train_log:
            if test_case[-1] == -1:
                success_imgs.append(img_list)
            else:
                if debug_reward_fun is not None and debug_check:
                    test_case[-1] = 'special'
                fail_imgs.append(img_list)


    IMG_IDX = 0

    # print('success num: {}'.format(success_num))
    tqdm.write('success num: {}'.format(success_num))
    # print('success id: {}'.format(success_id))

    actor.train()
    # return np.asarray(episode_rewards), test_case
    if train_log:
        return success_num, test_case, success_imgs, fail_imgs
    return success_num, test_case

@torch.no_grad()
def eval_actor_multi_push(
    env: gym.Env, actor_seq: List[nn.Module], device: str, n_episodes: int, seed: int, traj_limit: int, node_policy_dict: Dict, store_path=None, \
    final_rew_fun=None, loop_limit=None, train_log=False
) -> np.ndarray:
    debug_store = True

    if traj_limit is None:
        traj_limit = 150

    env.seed(seed)
    env.set_eval()
    for actor in actor_seq:
        actor.eval()

    episode_rewards = []

    # pdb.set_trace()
    # if store_path is None:
    #     store_path = "debug_imgs/eval"
    if store_path is not None:
        if not os.path.exists(store_path):
            os.mkdir(store_path)
    
    # pdb.set_trace()
    success_id = set()
    success_num = 0
    test_case = []
    if train_log:
        success_imgs = []
        fail_imgs = []
    for _ in trange(n_episodes):
        state, done = env.reset()[0], False
        episode_reward = 0.0
        episode_len = 0
        # debug
        test_case.append(0)

        reset_len = env.traj_len - env.traj_id
        loop_count = 0
        if train_log:
            img_list = []
        while (not done) and (episode_len <= min(traj_limit, reset_len)):
            if loop_limit is not None and loop_count > loop_limit:
                env.loop_reset(env.last_raw_obs)
                # pdb.set_trace()
                env.env.reset_gripper()
                loop_count = 0

            actor = actor_seq[node_policy_dict[env.cur_id]]
            action = actor.act(state, device) # / 2.0

            state, reward, terminate, truncate, info = env.step(action)

            # debug: test
            # print(rwd)
            if 'stage_done' in info and info['stage_done']:
                test_case[-1] += 1
                loop_count = 0
                # print('stage success')
                if env.cur_id == 0:
                    env.env.reset_gripper()
                    # terminate = True

            
            done = terminate or truncate
            
            if terminate and 'stage_done' in info and info['stage_done']:
                # final reward function (only for debug)
                if (not info['env_success']) and final_rew_fun is not None:
                    pdb.set_trace()
                    raw_obs = env.last_raw_obs
                    if final_rew_fun.get_reward(np.expand_dims(raw_obs, 0), 0):
                        test_case[-1] = 100
                        success_num += 1
                else:
                    test_case[-1] = 100
                    # print('episode success')
                    success_num += 1
                    # success_id.add(env.env.env.env._env.goal_idx)
                # test_case[-1] = 100
                # print('episode success')
                # success_num += 1

            episode_reward += reward
            episode_len += 1
            loop_count += 1
            img = env.render()
            if train_log:
                img_list.append(np.transpose(img, (2,0,1)))
            global IMG_IDX
            if store_path is not None:
                cv2.imwrite(f"{store_path}/{IMG_IDX}.png", img)
            # pdb.set_trace()
            IMG_IDX += 1
        # print(IMG_IDX)
        episode_rewards.append(episode_reward)
        if train_log:
            if test_case[-1] == 100:
                success_imgs.append(img_list)
            else:
                fail_imgs.append(img_list)


    IMG_IDX = 0

    # print('success num: {}'.format(success_num))
    tqdm.write('success num: {}'.format(success_num))
    # print('success id: {}'.format(success_id))

    actor.train()
    # return np.asarray(episode_rewards), test_case
    if train_log:
        return success_num, test_case, success_imgs, fail_imgs
    return success_num, test_case


@torch.no_grad()
def eval_actor_3(
    env: gym.Env, actors: List, device: str, n_episodes: int, seed: int, traj_limit: int
) -> np.ndarray:
    if traj_limit is None:
        traj_limit = 150

    env.seed(seed)
    env.set_eval()
    for actor in actors:
        actor.eval()

    episode_rewards = []

    # pdb.set_trace()
    if not os.path.exists("debug_imgs/eval"):
        os.mkdir("debug_imgs/eval")
    
    # pdb.set_trace()
    success_id = set()
    success_num = 0
    for e_id in trange(n_episodes):
        state, done = env.reset()[0], False
        crop_id = 1
        crop_time = 0
        episode_reward = 0.0
        episode_len = 0

        if not os.path.exists(f'debug_imgs/eval/{e_id}'):
            os.makedirs(f'debug_imgs/eval/{e_id}')

        reset_len = env.traj_len - env.traj_id
        while (not done) and (episode_len <= min(traj_limit, reset_len)):
            if env.cur_id == 0 or env.cur_id == 2:
                actor = actors[0]
            elif env.cur_id == 1 or env.cur_id == 3:
                actor = actors[1]
            else:
                print('what?')
                pdb.set_trace()

            action = actor.act(state, device) # / 2.0

            state, reward, terminate, truncate, info = env.step(action)

            # debug: test
            # print(rwd)
            if 'stage_done' in info and info['stage_done']:
                # pdb.set_trace()
                print('stage success (to {})'.format(env.crop_id))
                if crop_time == 1:
                    crop_id += 1
                    crop_time = 0
                else:
                    crop_time += 1
                # assert env.crop_id == crop_id
                if crop_id != 3 and crop_id != env.crop_id:
                    pdb.set_trace()
                if crop_id == 1:
                    if crop_time == 0:
                        assert env.cur_id == 0
                    elif crop_time == 1:
                        assert env.cur_id == 1
                elif crop_id == 1:
                    if crop_time == 0:
                        assert env.cur_id == 2
                    elif crop_time == 1:
                        assert env.cur_id == 3
            
            done = terminate or truncate
            
            if done and 'stage_done' in info and info['stage_done']:
                print('episode success')
                success_num += 1
                success_id.add(env.env.env.env._env.goal_idx)

            episode_reward += reward
            episode_len += 1
            img = env.render()
            global IMG_IDX
            cv2.imwrite(f"debug_imgs/eval/{e_id}/{IMG_IDX}_{env.crop_id}.png", img)
            IMG_IDX += 1
        episode_rewards.append(episode_reward)

    IMG_IDX = 0

    print('success num: {}'.format(success_num))
    print('success id: {}'.format(success_id))

    actor.train()
    return np.asarray(episode_rewards)

def define_policy(env, dataset, state_dim, action_dim, empty=False):
    config = TrainConfig()

    set_seed(config.seed, env, deterministic_torch=config.deterministic_torch)
    # set_env_seed(eval_env, config.eval_seed)

    if dataset is None:
        replay_buffer = None
    else:
        replay_buffer = ReplayBuffer(
            state_dim,
            action_dim,
            config.buffer_size,
            config.device,
        )
        # replay_buffer = ReplayBuffer(
        #     state_dim,
        #     action_dim,
        #     config.buffer_size,
        #     'cpu',
        # )
        if not empty:
            replay_buffer.load_d4rl_dataset(dataset)

    actor_critic_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.hidden_dim,
    }

    actor = Actor(**actor_critic_kwargs)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(config.device)
    critic_2.to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)

    awac = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        awac_lambda=config.awac_lambda,
    )
    return awac, replay_buffer

def define_deepset_policy(env, dataset, actor_state_dim, critic_state_dim, state_dim, action_dim, agg, obs_transit):
    config = TrainConfig()

    set_seed(config.seed, env, deterministic_torch=config.deterministic_torch)
    # set_env_seed(eval_env, config.eval_seed)

    if dataset is None:
        replay_buffer = None
    else:
        # replay_buffer = ReplayBuffer(
        #     state_dim,
        #     action_dim,
        #     config.buffer_size,
        #     config.device,
        # )
        replay_buffer = ReplayBuffer(
            state_dim,
            action_dim,
            config.buffer_size,
            'cpu',
        )
        replay_buffer.load_d4rl_dataset(dataset)

    actor_critic_kwargs = {
        "state_dim": actor_state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.hidden_dim,
        "agg": agg,
        "obs_transit": obs_transit
    }

    actor = DeepsetActor(**actor_critic_kwargs)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)

    actor_critic_kwargs = {
        "state_dim": critic_state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.hidden_dim,
        "obs_transit": obs_transit
    }
    critic_1 = DeepsetCritic(**actor_critic_kwargs)
    critic_2 = DeepsetCritic(**actor_critic_kwargs)
    critic_1.to(config.device)
    critic_2.to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)

    awac = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        awac_lambda=config.awac_lambda,
    )
    return awac, replay_buffer

def learn_policy(env, eval_env, awac: AdvantageWeightedActorCritic, replay_buffer: ReplayBuffer, traj_limit: int):
    config = TrainConfig()
    actor = awac._actor
    wandb_init(asdict(config))

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    state, _ = env.reset()

    print("Offline pretraining")
    for t in trange(
        int(config.offline_iterations) + int(config.online_iterations), ncols=80
    ):
        if t == config.offline_iterations:
            print("Online tuning")
        online_log = {}
        if t >= config.offline_iterations:
            action, _ = actor(
                torch.tensor(
                    state.reshape(1, -1), device=config.device, dtype=torch.float32
                )
            )
            action = action.cpu().data.numpy().flatten()
            next_state, reward, terminate, truncate, info = env.step(action)
            # add for skill environment
            if 'stage_done' in info:
                if info['stage_done']:
                    if terminate:
                        print('two block')
                    else:
                        print('one block')
                record_terminate = terminate or info['stage_done']
            else:
                record_terminate = terminate

            assert len(state.shape) == 1 and len(next_state.shape) == 1
            replay_buffer.add_transition(state, action, reward, next_state, record_terminate)
            
            # next
            state = next_state
            if terminate or truncate:
                state, _ = env.reset()
                # goal_achieved = False

        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        update_result = awac.update(batch)
        update_result[
            "offline_iter" if t < config.offline_iterations else "online_iter"
        ] = (t if t < config.offline_iterations else t - config.offline_iterations)
        update_result.update(online_log)
        wandb.log(update_result, step=t)
        if (t + 1) % config.eval_frequency == 0:
            eval_scores = eval_actor_scratch(
                eval_env, actor, device=config.device, n_episodes=config.n_test_episodes, seed=config.seed, traj_limit=traj_limit
            )
            eval_log = {"eval_reward": np.mean(eval_scores)}
            wandb.log(eval_log, step=t)
            
    wandb.finish()  
    
def learn_policy_multi(env, eval_env, awac_seq: List[AdvantageWeightedActorCritic], replay_buffer_seq: List[ReplayBuffer], traj_limit: int, node_policy_dict: Dict, \
                       update_all=False, store_path=None, env_name=None, loop_limit=None, non_terminate=False):
    config = TrainConfig()
    if env_name is not None:
       config.env_name = env_name 
    # actor = awac._actor
    wandb_init(asdict(config))
    best_score = 0

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    state, _ = env.reset()
    train_success_record = [0]
    train_reward_record = [0]
    
    episode_lens = [300]
    episode_len = 0
    episode_rewards = []

    # only for debug
    need_stop = False
    success_task = 0
    print("Offline pretraining")
    for t in trange(
        int(config.offline_iterations) + int(config.online_iterations), ncols=80
    ):
        if t == config.offline_iterations:
            print("Online tuning")
        online_log = {}
        if t >= config.offline_iterations:
            # find actor
            policy_id = node_policy_dict[env.cur_id]
            
            # only for debug
            # pdb.set_trace()
            # assert policy_id == 4

            awac = awac_seq[policy_id]
            actor = awac._actor
            replay_buffer = replay_buffer_seq[policy_id]
            # action
            action, _ = actor(
                torch.tensor(
                    state.reshape(1, -1), device=config.device, dtype=torch.float32
                )
            )
            action = action.cpu().data.numpy().flatten()

            # only for debug
            # img = env.render()
            # cv2.imwrite(f"store/debug_imgs/{episode_len}.png", img)

            next_state, reward, terminate, truncate, info = env.step(action)
            episode_rewards.append(reward)
            
            # add for skill environment
            if 'stage_done' in info:
                # if info['stage_done']:
                #     if terminate:
                #         print('two block')
                #     else:
                #         print('one block')
                record_terminate = terminate or info['stage_done']
            else:
                record_terminate = terminate

            # only for debug
            if record_terminate:
                success_task += 1
            #     need_stop = True
            #     pdb.set_trace()

            # never terminate for the final policy
            if non_terminate and terminate:
                record_terminate = False

            # just for record
            if terminate and 'stage_done' in info and info['stage_done']:
                if non_terminate and truncate:
                    # train_success_record.append(1)
                    train_success_record.append(success_task)
                    train_reward_record.append(np.mean(episode_rewards))
                elif not non_terminate:
                    # train_success_record.append(1)
                    train_success_record.append(success_task)
                    train_reward_record.append(np.mean(episode_rewards))
            elif terminate or truncate:
                # train_success_record.append(0)
                train_success_record.append(success_task)
                train_reward_record.append(np.mean(episode_rewards))

            assert len(state.shape) == 1 and len(next_state.shape) == 1
            replay_buffer.add_transition(state, action, reward, next_state, record_terminate)
            episode_len += 1

            # next
            state = next_state
            if non_terminate:
                if truncate:
                    state, _ = env.reset()
                    episode_lens.append(episode_len)
                    episode_len = 0
                    episode_rewards = []
                    success_task = 0
            elif terminate or truncate:
                state, _ = env.reset()
                episode_lens.append(episode_len)
                episode_len = 0
                episode_rewards = []
                success_task = 0
                # goal_achieved = False
                # if need_stop:
                #     pdb.set_trace()

        # online update
        if t < config.offline_iterations or update_all:
            awac_id = 0
            for awac, replay_buffer in zip(awac_seq, replay_buffer_seq):
                batch = replay_buffer.sample(config.batch_size)
                batch = [b.to(config.device) for b in batch]
                update_result = awac.update(batch)
                update_result[
                    "offline_iter" if t < config.offline_iterations else "online_iter"
                ] = (t if t < config.offline_iterations else t - config.offline_iterations)
                if awac_id == 0:
                    # update_result['train_reward'] = np.mean(train_success_record[-100:]).item()
                    update_result['train_reward'] = np.mean(train_reward_record[-100:]).item()
                    update_result['train_success_task'] = np.mean(train_success_record[-100:]).item()
                    update_result['episode_len'] = np.mean(episode_lens[-100:]).item()
                update_result.update(online_log)
                update_result = {k+f'_{awac_id}':update_result[k] for k in update_result}
                awac_id += 1
                wandb.log(update_result, step=t)
        else:
            batch = replay_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            update_result = awac.update(batch)
            update_result[
                "offline_iter" if t < config.offline_iterations else "online_iter"
            ] = (t if t < config.offline_iterations else t - config.offline_iterations)
            # update_result['train_reward'] = np.mean(train_success_record[-100:]).item()
            update_result['train_reward'] = np.mean(train_reward_record[-100:]).item()
            update_result['train_success_task'] = np.mean(train_success_record[-100:]).item()
            update_result['episode_len'] = np.mean(episode_lens[-100:]).item()
            update_result.update(online_log)
            update_result = {k+f'_{policy_id}':update_result[k] for k in update_result}
            wandb.log(update_result, step=t)
        if (t + 1) % config.eval_frequency == 0:
            if isinstance(eval_env, List):
                eval_log = {}
                for env_id, each_eval_env in enumerate(eval_env):
                    eval_scores, _, success_imgs, fail_imgs = eval_actor_multi(
                        each_eval_env, [awac._actor for awac in awac_seq], device=config.device, n_episodes=config.n_test_episodes, seed=config.seed, \
                        traj_limit=traj_limit, node_policy_dict=each_eval_env.node_policy_map, train_log=True, store_path=os.path.join(store_path, f'{env_id}_eval_img'),\
                        loop_limit=loop_limit
                    )
                    eval_log[f"eval_reward_{env_id}"] = eval_scores
                    # store images
                    # random.shuffle(success_imgs)
                    # random.shuffle(fail_imgs)
                    # for seq_id, img_seq in enumerate(success_imgs[:5]):
                    #     eval_log[f'{env_id}_success_sample_{seq_id}'] = wandb.Video(np.stack(img_seq, axis=0), fps=4)
                    # for seq_id, img_seq in enumerate(fail_imgs[:5]):
                    #     eval_log[f'{env_id}_fail_sample_{seq_id}'] = wandb.Video(np.stack(img_seq, axis=0), fps=4)
                    # store best
                    # if store_path is not None and env_id == 0 and eval_scores >= best_score:
                    #     best_score = eval_scores
                    #     for policy_id, policy in enumerate(awac_seq):
                    #         # policy.save(os.path.join(store_path, 'policy_{}.pth'.format(policy_id)))
                    #         policy.save(os.path.join(store_path, 'policy_{}_{}.pth'.format(policy_id, t)))
                for policy_id, policy in enumerate(awac_seq):
                    policy.save(os.path.join(store_path, 'policy_{}_{}.pth'.format(policy_id, t)))
            else:
                eval_log = {}
                eval_scores, _, success_imgs, fail_imgs = eval_actor_multi(
                    eval_env, [awac._actor for awac in awac_seq], device=config.device, n_episodes=config.n_test_episodes, seed=config.seed, \
                    traj_limit=traj_limit, node_policy_dict=node_policy_dict, train_log=True, store_path=os.path.join(store_path, f'eval_img'),\
                    loop_limit=loop_limit
                )
                eval_log = {"eval_reward": np.mean(eval_scores)}
                # eval_log = {"eval_reward": eval_scores}
                # store images
                # random.shuffle(success_imgs)
                # random.shuffle(fail_imgs)
                # for seq_id, img_seq in enumerate(success_imgs[:5]):
                #     eval_log[f'success_sample_{seq_id}'] = wandb.Video(np.stack(img_seq, axis=0), fps=15)
                # for seq_id, img_seq in enumerate(fail_imgs[:5]):
                #     eval_log[f'fail_sample_{seq_id}'] = wandb.Video(np.stack(img_seq, axis=0), fps=15)
                # store best
                # if store_path is not None and eval_scores >= best_score:
                #     best_score = eval_scores
                #     for policy_id, policy in enumerate(awac_seq):
                #         # policy.save(os.path.join(store_path, 'policy_{}.pth'.format(policy_id)))
                #         policy.save(os.path.join(store_path, 'policy_{}_{}.pth'.format(policy_id, t)))
                for policy_id, policy in enumerate(awac_seq):
                    policy.save(os.path.join(store_path, 'policy_{}_{}.pth'.format(policy_id, t)))
            # log
            wandb.log(eval_log, step=t)
            
    # wandb.finish()  
    
def learn_policy_multi_online(env, eval_env, awac_seq: List[AdvantageWeightedActorCritic], replay_buffer_seq: List[ReplayBuffer], traj_limit: int, node_policy_dict: Dict, \
                       update_all=False, store_path=None, env_name=None, loop_limit=None, learning_start=0):
    config = TrainConfig()
    if env_name is not None:
       config.env_name = env_name 
    # actor = awac._actor
    wandb_init(asdict(config))
    best_score = 0

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    state, _ = env.reset()
    train_success_record = [0]

    # print("Offline pretraining")
    print("Online tuning")
    for t in trange(int(config.online_iterations), ncols=80):
        online_log = {}
        # find actor
        policy_id = node_policy_dict[env.cur_id]
        awac = awac_seq[policy_id]
        actor = awac._actor
        replay_buffer = replay_buffer_seq[policy_id]
        # action
        action, _ = actor(
            torch.tensor(
                state.reshape(1, -1), device=config.device, dtype=torch.float32
            )
        )
        action = action.cpu().data.numpy().flatten()
        next_state, reward, terminate, truncate, info = env.step(action)
        
        # add for skill environment
        if 'stage_done' in info:
            # if info['stage_done']:
            #     if terminate:
            #         print('two block')
            #     else:
            #         print('one block')
            record_terminate = terminate or info['stage_done']
        else:
            record_terminate = terminate

        # just for record
        if terminate and 'stage_done' in info and info['stage_done']:
            train_success_record.append(1)
        elif terminate or truncate:
            train_success_record.append(0)

        assert len(state.shape) == 1 and len(next_state.shape) == 1
        replay_buffer.add_transition(state, action, reward, next_state, record_terminate)
        
        # next
        state = next_state
        if terminate or truncate:
            # pdb.set_trace()
            state, _ = env.reset()
            # goal_achieved = False

        # online update
        if t > learning_start:
            batch = replay_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            update_result = awac.update(batch)
            update_result[
                "offline_iter" if t < config.offline_iterations else "online_iter"
            ] = (t if t < config.offline_iterations else t - config.offline_iterations)
            update_result['train_reward'] = np.mean(train_success_record[-100:]).item()
            update_result.update(online_log)
            update_result = {k+f'_{policy_id}':update_result[k] for k in update_result}
            wandb.log(update_result, step=t)
        if (t + 1) % config.eval_frequency == 0:
            if isinstance(eval_env, List):
                eval_log = {}
                for policy_id, policy in enumerate(awac_seq):
                    policy.save(os.path.join(store_path, 'policy_{}_{}.pth'.format(policy_id, t)))
            else:
                eval_log = {}
                for policy_id, policy in enumerate(awac_seq):
                    policy.save(os.path.join(store_path, 'policy_{}_{}.pth'.format(policy_id, t)))
            # log
            wandb.log(eval_log, step=t)
            
    wandb.finish()  
    
def learn_policy_multi_debug(env, eval_env, awac_seq: List[AdvantageWeightedActorCritic], replay_buffer_seq: List[ReplayBuffer], \
                             reward_fun_seq: List, traj_limit: int, node_policy_dict: Dict, \
                             update_all=False, store_path=None, env_name=None, loop_limit=None, non_terminate=False):
    config = TrainConfig()
    if env_name is not None:
       config.env_name = env_name 
    # actor = awac._actor
    # wandb_init(asdict(config))
    best_score = 0

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    state, _ = env.reset()
    train_success_record = [0]
    episode_lens = [300]
    episode_len = 0

    # only for debug
    need_stop = False

    print("Offline pretraining")
    for t in trange(
        int(config.offline_iterations) + int(config.online_iterations), ncols=80
    ):
        if t == config.offline_iterations:
            print("Online tuning")
        online_log = {}
        if t >= config.offline_iterations:
            # find actor
            policy_id = node_policy_dict[env.cur_id]
            
            # only for debug
            # assert policy_id == 4

            awac = awac_seq[policy_id]
            actor = awac._actor
            replay_buffer = replay_buffer_seq[policy_id]
            # action
            action, _ = actor(
                torch.tensor(
                    state.reshape(1, -1), device=config.device, dtype=torch.float32
                )
            )
            action = action.cpu().data.numpy().flatten()

            # only for debug
            img = env.render()
            cv2.imwrite(f"store/debug_imgs/{episode_len}.png", img)

            next_state, reward, terminate, truncate, info = env.step(action)
            
            # add for skill environment
            if 'stage_done' in info:
                # if info['stage_done']:
                #     if terminate:
                #         print('two block')
                #     else:
                #         print('one block')
                record_terminate = terminate or info['stage_done']
            else:
                record_terminate = terminate

            # only for debug
            if record_terminate:
                need_stop = True
                pdb.set_trace()

            # never terminate for the final policy
            if non_terminate and terminate:
                record_terminate = False

            # just for record
            if terminate and 'stage_done' in info and info['stage_done']:
                if non_terminate and truncate:
                    train_success_record.append(1)
                elif not non_terminate:
                    train_success_record.append(1)
            elif terminate or truncate:
                train_success_record.append(0)

            assert len(state.shape) == 1 and len(next_state.shape) == 1
            replay_buffer.add_transition(state, action, reward, next_state, record_terminate)
            episode_len += 1

            # next
            state = next_state
            if non_terminate:
                if truncate:
                    state, _ = env.reset()
                    episode_lens.append(episode_len)
                    episode_len = 0
            elif terminate or truncate:
                state, _ = env.reset()
                episode_lens.append(episode_len)
                episode_len = 0
                # goal_achieved = False
                if need_stop:
                    pdb.set_trace()

        # online update
        if t < config.offline_iterations or update_all:
            awac_id = 0
            for awac, replay_buffer in zip(awac_seq, replay_buffer_seq):
                batch = replay_buffer.sample(config.batch_size)
                batch = [b.to(config.device) for b in batch]
                update_result = awac.update(batch)
                # update_result[
                #     "offline_iter" if t < config.offline_iterations else "online_iter"
                # ] = (t if t < config.offline_iterations else t - config.offline_iterations)
                # if awac_id == 0:
                #     update_result['train_reward'] = np.mean(train_success_record[-100:]).item()
                #     update_result['episode_len'] = np.mean(episode_lens[-100:]).item()
                # update_result.update(online_log)
                # update_result = {k+f'_{awac_id}':update_result[k] for k in update_result}
                awac_id += 1
                # wandb.log(update_result, step=t)
        else:
            batch = replay_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            update_result = awac.update(batch)
            # update_result[
            #     "offline_iter" if t < config.offline_iterations else "online_iter"
            # ] = (t if t < config.offline_iterations else t - config.offline_iterations)
            # update_result['train_reward'] = np.mean(train_success_record[-100:]).item()
            # update_result['episode_len'] = np.mean(episode_lens[-100:]).item()
            # update_result.update(online_log)
            # update_result = {k+f'_{policy_id}':update_result[k] for k in update_result}
            # wandb.log(update_result, step=t)
        if (t + 1) % config.eval_frequency == 0:
            if isinstance(eval_env, List):
                eval_log = {}
                # for env_id, each_eval_env in enumerate(eval_env):
                    # eval_scores, _, success_imgs, fail_imgs = eval_actor_multi(
                    #     each_eval_env, [awac._actor for awac in awac_seq], device=config.device, n_episodes=config.n_test_episodes, seed=config.seed, \
                    #     traj_limit=traj_limit, node_policy_dict=each_eval_env.node_policy_map, train_log=True, store_path=os.path.join(store_path, f'{env_id}_eval_img'),\
                    #     loop_limit=loop_limit
                    # )
                    # eval_log[f"eval_reward_{env_id}"] = eval_scores
                    # store images
                    # random.shuffle(success_imgs)
                    # random.shuffle(fail_imgs)
                    # for seq_id, img_seq in enumerate(success_imgs[:5]):
                    #     eval_log[f'{env_id}_success_sample_{seq_id}'] = wandb.Video(np.stack(img_seq, axis=0), fps=4)
                    # for seq_id, img_seq in enumerate(fail_imgs[:5]):
                    #     eval_log[f'{env_id}_fail_sample_{seq_id}'] = wandb.Video(np.stack(img_seq, axis=0), fps=4)
                    # store best
                    # if store_path is not None and env_id == 0 and eval_scores >= best_score:
                    #     best_score = eval_scores
                    #     for policy_id, policy in enumerate(awac_seq):
                    #         # policy.save(os.path.join(store_path, 'policy_{}.pth'.format(policy_id)))
                    #         policy.save(os.path.join(store_path, 'policy_{}_{}.pth'.format(policy_id, t)))
                for policy_id, policy in enumerate(awac_seq):
                    policy.save(os.path.join(store_path, 'policy_{}_{}.pth'.format(policy_id, t)))
            else:
                eval_log = {}
                # eval_scores, _, success_imgs, fail_imgs = eval_actor_multi(
                #     eval_env, [awac._actor for awac in awac_seq], device=config.device, n_episodes=config.n_test_episodes, seed=config.seed, \
                #     traj_limit=traj_limit, node_policy_dict=node_policy_dict, train_log=True, store_path=os.path.join(store_path, f'eval_img'),\
                #     loop_limit=loop_limit
                # )
                # eval_log = {"eval_reward": np.mean(eval_scores)}
                # eval_log = {"eval_reward": eval_scores}
                # store images
                # random.shuffle(success_imgs)
                # random.shuffle(fail_imgs)
                # for seq_id, img_seq in enumerate(success_imgs[:5]):
                #     eval_log[f'success_sample_{seq_id}'] = wandb.Video(np.stack(img_seq, axis=0), fps=15)
                # for seq_id, img_seq in enumerate(fail_imgs[:5]):
                #     eval_log[f'fail_sample_{seq_id}'] = wandb.Video(np.stack(img_seq, axis=0), fps=15)
                # store best
                # if store_path is not None and eval_scores >= best_score:
                #     best_score = eval_scores
                #     for policy_id, policy in enumerate(awac_seq):
                #         # policy.save(os.path.join(store_path, 'policy_{}.pth'.format(policy_id)))
                #         policy.save(os.path.join(store_path, 'policy_{}_{}.pth'.format(policy_id, t)))
                for policy_id, policy in enumerate(awac_seq):
                    policy.save(os.path.join(store_path, 'policy_{}_{}.pth'.format(policy_id, t)))
            # log
            # wandb.log(eval_log, step=t)
            
    # wandb.finish()  

def train_block(config: TrainConfig, dataset, env, obs_transit, all_rew_fun, add_rew_fun):
    is_env_with_goal = False

    max_steps = 100

    set_seed(config.seed, env, deterministic_torch=config.deterministic_torch)
    # set_env_seed(eval_env, config.eval_seed)
    state_dim = 16
    action_dim = 4

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(dataset, config.env_name)

    # state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    state_mean, state_std = 0, 1
    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    # env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    # eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    actor_critic_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.hidden_dim,
    }

    actor = Actor(**actor_critic_kwargs)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(config.device)
    critic_2.to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)

    awac = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        awac_lambda=config.awac_lambda,
    )
    wandb_init(asdict(config))

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    full_eval_scores, full_normalized_eval_scores = [], []
    state, done = env.reset()[0], False
    episode_step = 0
    episode_return = 0
    goal_achieved = False

    eval_successes = []
    train_successes = []

    crop_id = 1
    state = obs_transit.do_crop_state(np.array([state]), crop_id)
    print("Offline pretraining")
    for t in trange(
        int(config.offline_iterations) + int(config.online_iterations), ncols=80
    ):
        if t == config.offline_iterations:
            print("Online tuning")
        online_log = {}
        if t >= config.offline_iterations:
            episode_step += 1
            action, _ = actor(
                torch.tensor(
                    state.reshape(1, -1), device=config.device, dtype=torch.float32
                )
            )
            action = action.cpu().data.numpy().flatten()
            next_state, reward, terminate, truncate, info = env.step(action)
            rwd, cost = all_rew_fun[crop_id - 1].get_reward(next_state, -1)
            rwd = add_rew_fun(rwd, cost, all_rew_fun[crop_id - 1], False)
            if rwd > 0.9:
                print("one block succeed")
                if crop_id == 2:
                    print("two block success")
                    done = True
                crop_id = 2

            next_state = obs_transit.do_crop_state(np.array([next_state]), crop_id)
            replay_buffer.add_transition(state[0], action, rwd, next_state[0], rwd > 0.9)
            state = next_state
            if episode_step >= max_steps:
                done = True
            if done:
                # pdb.set_trace()
                state, done = env.reset()[0], False
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                # if is_env_with_goal:
                    # train_successes.append(goal_achieved)
                    # online_log["train/regret"] = np.mean(1 - np.array(train_successes))
                    # online_log["train/is_success"] = float(goal_achieved)
                # online_log["train/episode_return"] = episode_return
                # normalized_return = eval_env.get_normalized_score(episode_return)
                # online_log["train/d4rl_normalized_episode_return"] = (
                    # normalized_return * 100.0
                # )
                # online_log["train/episode_length"] = episode_step
                episode_return = 0
                episode_step = 0
                crop_id = 1
                state = obs_transit.do_crop_state(np.array([state]), crop_id)
                # goal_achieved = False

        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        update_result = awac.update(batch)
        update_result[
            "offline_iter" if t < config.offline_iterations else "online_iter"
        ] = (t if t < config.offline_iterations else t - config.offline_iterations)
        update_result.update(online_log)
        wandb.log(update_result, step=t)
        if (t + 1) % config.eval_frequency == 0:
            eval_scores = eval_actor(
                dataset, env, obs_transit, all_rew_fun, add_rew_fun, state_mean, state_std, actor, device=config.device, n_episodes=config.n_test_episodes, seed=config.seed
            )
            eval_log = {}

            # full_eval_scores.append(eval_scores)
            # wandb.log({"eval/eval_score": eval_scores.mean()}, step=t)
            # if hasattr(eval_env, "get_normalized_score"):
            #     normalized = eval_env.get_normalized_score(np.mean(eval_scores))
            #     # Valid only for envs with goal, e.g. AntMaze, Adroit
            #     if t >= config.offline_iterations and is_env_with_goal:
            #         eval_successes.append(success_rate)
            #         eval_log["eval/regret"] = np.mean(1 - np.array(train_successes))
            #         eval_log["eval/success_rate"] = success_rate
            #     normalized_eval_scores = normalized * 100.0
            #     full_normalized_eval_scores.append(normalized_eval_scores)
            #     eval_log["eval/d4rl_normalized_score"] = normalized_eval_scores
            #     wandb.log(eval_log, step=t)
            # if config.checkpoints_path:
            #     torch.save(
            #         awac.state_dict(),
            #         os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
            #     )
    wandb.finish()

def train_block_2(config: TrainConfig, dataset, env, eval_env, obs_transit, all_rew_fun, add_rew_fun):
    is_env_with_goal = False

    max_steps = 100

    set_seed(config.seed, env, deterministic_torch=config.deterministic_torch)
    # set_env_seed(eval_env, config.eval_seed)
    state_dim = 16
    action_dim = 4

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(dataset, config.env_name)

    awac, replay_buffer = define_policy(env, dataset, 16, 4)
    actor = awac._actor

    # state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    state_mean, state_std = 0, 1
    wandb_init(asdict(config))

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    full_eval_scores, full_normalized_eval_scores = [], []
    state, done = env.reset()[0], False
    episode_step = 0
    episode_return = 0
    goal_achieved = False

    eval_successes = []
    train_successes = []

    crop_id = 1
    # state = obs_transit.do_crop_state(np.array([state]), crop_id)
    print("Offline pretraining")
    for t in trange(
        int(config.offline_iterations) + int(config.online_iterations), ncols=80
    ):
        if t == config.offline_iterations:
            print("Online tuning")
        online_log = {}
        if t >= config.offline_iterations:
            episode_step += 1
            action, _ = actor(
                torch.tensor(
                    state.reshape(1, -1), device=config.device, dtype=torch.float32
                )
            )
            action = action.cpu().data.numpy().flatten()
            next_state, reward, terminate, truncate, info = env.step(action)

            # if reward > 0:
            #     pdb.set_trace()

            # add for skill environment
            if 'stage_done' in info:
                if reward > 0.9:
                    assert info['stage_done']
                if info['stage_done']:
                    assert reward > 0.9

                record_terminate = terminate or info['stage_done']
                if info['stage_done']:
                    if terminate:
                        done = True
                        print("two blocks")
                        # pdb.set_trace()
                    else:
                        print("one block")
                        assert env.crop_id == 2
            else:
                record_terminate = terminate
            if record_terminate:
                # pdb.set_trace()
                assert reward > 0.9

            # double check
            assert len(state.shape) == 1 and len(next_state.shape) == 1
            replay_buffer.add_transition(state, action, reward, next_state, record_terminate)


            # next_state, reward, terminate, truncate, info = env.step(action)
            # rwd, cost = all_rew_fun[crop_id - 1].get_reward(next_state, -1)
            # rwd = add_rew_fun(rwd, cost, all_rew_fun[crop_id - 1], False)
            # if rwd > 0.9:
            #     print("one block succeed")
            #     if crop_id == 2:
            #         done = True
            #     crop_id = 2

            # next_state = obs_transit.do_crop_state(np.array([next_state]), crop_id)
            # replay_buffer.add_transition(state[0], action, rwd, next_state[0], rwd > 0.9)
            state = next_state
            if episode_step >= max_steps:
                done = True
            # if terminate or truncate:
            #     done = True
            if done:
                # pdb.set_trace()
                state, done = env.reset()[0], False
                # state, done = env.reset()
                assert env.crop_id == 1

                episode_return = 0
                episode_step = 0
                crop_id = 1
                # state = obs_transit.do_crop_state(np.array([state]), crop_id)
                # goal_achieved = False

        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        update_result = awac.update(batch)
        update_result[
            "offline_iter" if t < config.offline_iterations else "online_iter"
        ] = (t if t < config.offline_iterations else t - config.offline_iterations)
        update_result.update(online_log)
        wandb.log(update_result, step=t)
        if (t + 1) % config.eval_frequency == 0:
            # eval_scores = eval_actor(
            #     dataset, eval_env.env, obs_transit, all_rew_fun, add_rew_fun, state_mean, state_std, actor, device=config.device, n_episodes=config.n_test_episodes, seed=config.seed
            # )
            eval_scores = eval_actor_scratch(
                eval_env, actor, device=config.device, n_episodes=config.n_test_episodes, seed=config.seed, traj_limit=150
            )
            eval_log = {}

    wandb.finish()

def train_block_3(config: TrainConfig, dataset, env, eval_env, obs_transit, all_rew_fun, add_rew_fun):
    is_env_with_goal = False

    max_steps = 150

    set_seed(config.seed, env, deterministic_torch=config.deterministic_torch)
    # set_env_seed(eval_env, config.eval_seed)
    state_dim = 16
    action_dim = 4

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(dataset, config.env_name)

    awac_1, replay_buffer_1 = define_policy(env, dataset[0], 16, 4)
    actor_1 = awac_1._actor

    awac_2, replay_buffer_2 = define_policy(env, dataset[1], 16, 4)
    actor_2 = awac_2._actor

    # state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    state_mean, state_std = 0, 1
    wandb_init(asdict(config))

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    full_eval_scores, full_normalized_eval_scores = [], []
    state, done = env.reset()[0], False
    episode_step = 0
    episode_return = 0
    goal_achieved = False

    eval_successes = []
    train_successes = []

    crop_id = 1
    crop_time = 0
    # state = obs_transit.do_crop_state(np.array([state]), crop_id)
    print("Offline pretraining")
    for t in trange(
        int(config.offline_iterations) + int(config.online_iterations), ncols=80
    ):
        if t == config.offline_iterations:
            print("Online tuning")
        online_log = {}
        if t >= config.offline_iterations:
            episode_step += 1
            # find actor
            if env.cur_id == 0 or env.cur_id == 2:
                actor = actor_1
                replay_buffer = replay_buffer_1
            elif env.cur_id == 1 or env.cur_id == 3:
                actor = actor_2
                replay_buffer = replay_buffer_2
            else:
                print('what?')

            action, _ = actor(
                torch.tensor(
                    state.reshape(1, -1), device=config.device, dtype=torch.float32
                )
            )
            action = action.cpu().data.numpy().flatten()
            next_state, reward, terminate, truncate, info = env.step(action)

            # add for skill environment
            if 'stage_done' in info:
                if info['stage_done']:
                    if crop_time == 1:
                        crop_id += 1
                        crop_time = 0
                    else:
                        crop_time += 1

                if crop_id != 3 and crop_id != env.crop_id:
                    pdb.set_trace()
                if crop_id == 1:
                    if crop_time == 0:
                        assert env.cur_id == 0
                    elif crop_time == 1:
                        try:
                            assert env.cur_id == 1
                        except:
                            pdb.set_trace()
                elif crop_id == 2:
                    if crop_time == 0:
                        assert env.cur_id == 2
                    elif crop_time == 1:
                        assert env.cur_id == 3
                # assert crop_id == env.crop_id

                if reward > 0.9:
                    assert info['stage_done']
                if info['stage_done']:
                    assert reward > 0.9

                record_terminate = terminate or info['stage_done']
                if info['stage_done']:
                    if terminate:
                        done = True
                        print("all done")
                    else:
                        print("{} id finish".format(env.crop_id))
            else:
                record_terminate = terminate
            if record_terminate:
                # pdb.set_trace()
                assert reward > 0.9

            # double check
            assert len(state.shape) == 1 and len(next_state.shape) == 1
            replay_buffer.add_transition(state, action, reward, next_state, record_terminate)

            state = next_state
            if episode_step >= max_steps:
                done = True
            # if terminate or truncate:
            #     done = True
            if done:
                # pdb.set_trace()
                state, done = env.reset()[0], False
                # state, done = env.reset()
                assert env.crop_id == 1
                episode_step = 0
                crop_id = 1
                crop_time = 0
                # state = obs_transit.do_crop_state(np.array([state]), crop_id)
                # goal_achieved = False

        # train 1
        batch = replay_buffer_1.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        update_result = awac_1.update(batch)
        update_result[
            "offline_iter" if t < config.offline_iterations else "online_iter"
        ] = (t if t < config.offline_iterations else t - config.offline_iterations)
        update_result.update(online_log)

        # train 2
        batch = replay_buffer_2.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        update_result = awac_2.update(batch)
        update_result[
            "offline_iter" if t < config.offline_iterations else "online_iter"
        ] = (t if t < config.offline_iterations else t - config.offline_iterations)
        update_result.update(online_log)

        wandb.log(update_result, step=t)
        if (t + 1) % config.eval_frequency == 0:
            # eval_scores = eval_actor(
            #     dataset, eval_env.env, obs_transit, all_rew_fun, add_rew_fun, state_mean, state_std, actor, device=config.device, n_episodes=config.n_test_episodes, seed=config.seed
            # )
            eval_scores = eval_actor_3(
                eval_env, [actor_1, actor_2], device=config.device, n_episodes=config.n_test_episodes, seed=config.seed, traj_limit=150
            )
            eval_log = {}

    wandb.finish()


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)

    is_env_with_goal = config.env_name.startswith(ENVS_WITH_GOAL)

    max_steps = env._max_episode_steps

    set_seed(config.seed, env, deterministic_torch=config.deterministic_torch)
    set_env_seed(eval_env, config.eval_seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset = d4rl.qlearning_dataset(env)

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(dataset, config.env_name)

    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    actor_critic_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.hidden_dim,
    }

    actor = Actor(**actor_critic_kwargs)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(config.device)
    critic_2.to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)

    awac = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        awac_lambda=config.awac_lambda,
    )
    wandb_init(asdict(config))

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    full_eval_scores, full_normalized_eval_scores = [], []
    state, done = env.reset(), False
    episode_step = 0
    episode_return = 0
    goal_achieved = False

    eval_successes = []
    train_successes = []

    print("Offline pretraining")
    for t in trange(
        int(config.offline_iterations) + int(config.online_iterations), ncols=80
    ):
        if t == config.offline_iterations:
            print("Online tuning")
        online_log = {}
        if t >= config.offline_iterations:
            episode_step += 1
            action, _ = actor(
                torch.tensor(
                    state.reshape(1, -1), device=config.device, dtype=torch.float32
                )
            )
            action = action.cpu().data.numpy().flatten()
            next_state, reward, done, env_infos = env.step(action)

            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
            episode_return += reward
            real_done = False  # Episode can timeout which is different from done
            if done and episode_step < max_steps:
                real_done = True

            if config.normalize_reward:
                reward = modify_reward_online(reward, config.env_name, **reward_mod_dict)

            replay_buffer.add_transition(state, action, reward, next_state, real_done)
            state = next_state
            if done:
                state, done = env.reset(), False
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                if is_env_with_goal:
                    train_successes.append(goal_achieved)
                    online_log["train/regret"] = np.mean(1 - np.array(train_successes))
                    online_log["train/is_success"] = float(goal_achieved)
                online_log["train/episode_return"] = episode_return
                normalized_return = eval_env.get_normalized_score(episode_return)
                online_log["train/d4rl_normalized_episode_return"] = (
                    normalized_return * 100.0
                )
                online_log["train/episode_length"] = episode_step
                episode_return = 0
                episode_step = 0
                goal_achieved = False

        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        update_result = awac.update(batch)
        update_result[
            "offline_iter" if t < config.offline_iterations else "online_iter"
        ] = (t if t < config.offline_iterations else t - config.offline_iterations)
        update_result.update(online_log)
        wandb.log(update_result, step=t)
        if (t + 1) % config.eval_frequency == 0:
            eval_scores, success_rate = eval_actor(
                eval_env, actor, config.device, config.n_test_episodes, config.test_seed
            )
            eval_log = {}

            full_eval_scores.append(eval_scores)
            wandb.log({"eval/eval_score": eval_scores.mean()}, step=t)
            if hasattr(eval_env, "get_normalized_score"):
                normalized = eval_env.get_normalized_score(np.mean(eval_scores))
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                if t >= config.offline_iterations and is_env_with_goal:
                    eval_successes.append(success_rate)
                    eval_log["eval/regret"] = np.mean(1 - np.array(train_successes))
                    eval_log["eval/success_rate"] = success_rate
                normalized_eval_scores = normalized * 100.0
                full_normalized_eval_scores.append(normalized_eval_scores)
                eval_log["eval/d4rl_normalized_score"] = normalized_eval_scores
                wandb.log(eval_log, step=t)
            if config.checkpoints_path:
                torch.save(
                    awac.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
    wandb.finish()


if __name__ == "__main__":
    train()