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
from tqdm import trange

TensorBatch = List[torch.Tensor]

ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")

import cv2
import pdb

IMG_IDX = 0

def set_env_seed(env: Optional[gym.Env], seed: int):
    env.seed(seed)
    env.action_space.seed(seed)


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
def eval_actor_multi(
    env: gym.Env, actor_seq: List[nn.Module], device: str, n_episodes: int, seed: int, traj_limit: int, node_policy_dict: Dict, store_path=None, final_rew_fun=None, loop_limit=None,
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
    if store_path is None:
        store_path = "debug_imgs/eval"
    if store_path is not None:
        if not os.path.exists(store_path):
            os.mkdir(store_path)
    
    # pdb.set_trace()
    success_id = set()
    success_num = 0
    test_case = []
    for _ in trange(n_episodes):
        state, done = env.reset()[0], False
        episode_reward = 0.0
        episode_len = 0
        # debug
        test_case.append(0)

        reset_len = env.traj_len - env.traj_id
        loop_count = 0
        while (not done) and (episode_len <= min(traj_limit, reset_len)):
            if loop_limit is not None and loop_count > loop_limit:
                env.loop_reset(env.last_raw_obs)
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

            episode_reward += reward
            episode_len += 1
            loop_count += 1
            img = env.render()
            global IMG_IDX
            if store_path is not None:
                cv2.imwrite(f"{store_path}/{IMG_IDX}.png", img)
            IMG_IDX += 1
        # print(IMG_IDX)
        episode_rewards.append(episode_reward)

    IMG_IDX = 0

    print('success num: {}'.format(success_num))
    # print('success id: {}'.format(success_id))

    actor.train()
    # return np.asarray(episode_rewards), test_case
    return success_num, test_case


def learn_policy_multi(env, eval_env, policy_seq, traj_limit: int, node_policy_dict: Dict, update_all=False, store_path=None):
    config = TrainConfig()
    # actor = awac._actor
    wandb_init(asdict(config))
    best_score = 0

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

            assert len(state.shape) == 1 and len(next_state.shape) == 1
            replay_buffer.add_transition(state, action, reward, next_state, record_terminate)
            
            # next
            state = next_state
            if terminate or truncate:
                state, _ = env.reset()
                # goal_achieved = False

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
            update_result.update(online_log)
            update_result = {k+f'_{policy_id}':update_result[k] for k in update_result}
            wandb.log(update_result, step=t)
        if (t + 1) % config.eval_frequency == 0:
            if isinstance(eval_env, List):
                eval_log = {}
                for env_id, each_eval_env in enumerate(eval_env):
                    eval_scores, _ = eval_actor_multi(
                        each_eval_env, [awac._actor for awac in awac_seq], device=config.device, n_episodes=config.n_test_episodes, seed=config.seed, traj_limit=traj_limit, node_policy_dict=each_eval_env.node_policy_map
                    )
                    eval_log[f"eval_reward_{env_id}"] = eval_scores
                    # store best
                    if store_path is not None and env_id == 0 and eval_scores > best_score:
                        best_score = eval_scores
                        for policy_id, policy in enumerate(awac_seq):
                            policy.save(os.path.join(store_path, 'policy_{}.pth'.format(policy_id)))
            else:
                eval_scores, _ = eval_actor_multi(
                    eval_env, [awac._actor for awac in awac_seq], device=config.device, n_episodes=config.n_test_episodes, seed=config.seed, traj_limit=traj_limit, node_policy_dict=node_policy_dict
                )
                # eval_log = {"eval_reward": np.mean(eval_scores)}
                eval_log = {"eval_reward": eval_scores}
                # store best
                if store_path is not None and eval_scores > best_score:
                    best_score = eval_scores
                    for policy_id, policy in enumerate(awac_seq):
                        policy.save(os.path.join(store_path, 'policy_{}.pth'.format(policy_id)))
            # log
            wandb.log(eval_log, step=t)
            
    wandb.finish()  
    


if __name__ == "__main__":
    train()