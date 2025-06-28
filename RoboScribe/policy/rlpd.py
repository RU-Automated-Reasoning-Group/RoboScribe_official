import os
import pickle
import wandb
import tqdm
import numpy as np
import sys
import gym

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags
from ml_collections.config_dict import config_dict

from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer
from rlpd.data.d4rl_datasets import D4RLDataset
from absl import flags

try:
    from rlpd.data.binary_datasets import BinaryDataset
except:
    print("not importing binary dataset")
from rlpd.evaluation import evaluate, evaluate_single, evaluate_multi
from rlpd.wrappers import wrap_gym

import pdb


def get_flags():
    FLAGS = flags.FLAGS

    flags.DEFINE_string("project_name", "rlpd", "wandb project name.")
    flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
    flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
    flags.DEFINE_integer("seed", 42, "Random seed.")
    flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
    flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
    flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
    flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
    flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
    flags.DEFINE_integer(
        "start_training", int(1e4), "Number of training steps to start training."
    )
    flags.DEFINE_integer("pretrain_steps", 0, "Number of offline updates.")
    flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
    flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
    flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
    flags.DEFINE_boolean(
        "checkpoint_buffer", False, "Save agent replay buffer on evaluation."
    )
    flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
    flags.DEFINE_boolean(
        "binary_include_bc", True, "Whether to include BC data in the binary datasets."
    )

    config_flags.DEFINE_config_file(
        "config",
        "rlpd/configs/sac_config.py",
        "File path to the training hyperparameter configuration.",
        lock_config=False,
    )

    FLAGS(sys.argv)

    return FLAGS

class TrainConfig:
    # learning rate
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_dims = (256, 256)
    discount: float = 0.99
    num_qs: int = 10
    num_min_qs: int = 2
    tau: float = 0.005
    critic_layer_norm: bool = True
    model_cls: str = "SACLearner"
    temp_lr: float = 3e-4
    init_temperature: float = 1.0
    # target_entropy = config_dict.placeholder(float)
    target_entropy = None
    backup_entropy = True
    # critic_weight_decay = config_dict.placeholder(float)
    critic_weight_decay = None

    project_name: str = 'rlpd'
    offline_ratio: float = 0.5
    seed: int = 42
    eval_episodes: int = 10
    log_interval: int = 1000
    eval_interval: int = 40000
    batch_size: int = 256
    max_steps: int = int(2e7)
    start_training: int = 1e4
    pretrain_steps: int = 0
    tqdm: bool = True
    save_video: bool = True
    checkpoint_model: bool = True
    checkpoint_buffer: bool = False
    utd_ratio: int = 1
    binary_include_bc: bool = True

class TrainConfigSingle:
    # learning rate
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_dims = (256, 256)
    discount: float = 0.99
    num_qs: int = 10
    num_min_qs: int = 2
    tau: float = 0.005
    critic_layer_norm: bool = True
    model_cls: str = "SACLearner"
    temp_lr: float = 3e-4
    init_temperature: float = 1.0
    # target_entropy = config_dict.placeholder(float)
    target_entropy = None
    backup_entropy = True
    # critic_weight_decay = config_dict.placeholder(float)
    critic_weight_decay = None

    project_name: str = 'rlpdSingle'
    offline_ratio: float = 0.5
    seed: int = 42
    eval_episodes: int = 20
    log_interval: int = 1000
    eval_interval: int = 20000
    batch_size: int = 256
    max_steps: int = int(8e5)
    # max_steps: int = int(8e4)
    start_training: int = 1e4
    pretrain_steps: int = 0
    tqdm: bool = True
    save_video: bool = False
    checkpoint_model: bool = True
    checkpoint_buffer: bool = False
    utd_ratio: int = 1
    binary_include_bc: bool = True

def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return combined

def to_gym_space(observation_space, action_space):
    new_obs_space = gym.spaces.Box(low=observation_space.low, high=observation_space.high, shape=observation_space.shape, dtype=observation_space.dtype)
    new_act_space = gym.spaces.Box(low=action_space.low, high=action_space.high, shape=action_space.shape, dtype=action_space.dtype)

    return new_obs_space, new_act_space

def learn_policy_multi(env, eval_env, node_policy_dict, demo_dataset_dict, update_all=False, 
                       store_path=None):
    
    # FLAGS = get_flags()
    FLAGS = TrainConfig
    flag_var = vars(FLAGS)
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    wandb.init(project=FLAGS.project_name)
    wandb.config.update({k:flag_var[k] for k in flag_var if '__' not in k})

    # not ideal, but works for now:
    # if "binary" in FLAGS.env_name:
    #     ds = BinaryDataset(env, include_bc_data=FLAGS.binary_include_bc)
    # else:
    #     ds = D4RLDataset(env)

    # init
    kwargs = {k:flag_var[k] for k in flag_var if '__' not in k}
    model_cls = kwargs.pop("model_cls")
    policy_ids = set(node_policy_dict.values())
    policy_dict = {}
    replay_buffer_dict = {}

    # init for record
    success_task = 0
    episode_rewards = []
    observation_space, action_space = to_gym_space(env.observation_space, env.action_space)

    # get policy and replay buffer
    for p_id in policy_ids:
        policy_dict[p_id] = globals()[model_cls].create(
                                FLAGS.seed, observation_space, action_space, actor_lr=FLAGS.actor_lr, critic_lr=FLAGS.critic_lr,
                                hidden_dims=FLAGS.hidden_dims, discount=FLAGS.discount, num_qs=FLAGS.num_qs, num_min_qs=FLAGS.num_min_qs, tau=FLAGS.tau, 
                                critic_layer_norm=FLAGS.critic_layer_norm, temp_lr=FLAGS.temp_lr, init_temperature=FLAGS.init_temperature,
                                target_entropy=FLAGS.target_entropy, backup_entropy=FLAGS.backup_entropy, critic_weight_decay=FLAGS.critic_weight_decay
                            )
        replay_buffer_dict[p_id] = ReplayBuffer(
                                observation_space, action_space, FLAGS.max_steps
                            )
        replay_buffer_dict[p_id].seed(FLAGS.seed)

    # offline training
    for i in tqdm.tqdm(
        range(0, FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        for p_id in policy_ids:
            offline_batch = demo_dataset_dict[p_id].sample(FLAGS.batch_size * FLAGS.utd_ratio)
            batch = {}
            for k, v in offline_batch.items():
                batch[k] = v

            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"offline-training/{k}": v}, step=i)

            if i % FLAGS.eval_interval == 0:
                eval_info = evaluate_multi(policy_dict, eval_env, num_episodes=FLAGS.eval_episodes, node_policy_dict=node_policy_dict)
                for k, v in eval_info.items():
                    wandb.log({f"offline-evaluation/{k}": v}, step=i)

    # online training
    observation, _ = env.reset()
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        # find actor and replay buffer
        cur_policy_id = node_policy_dict[env.cur_id]
        agent = policy_dict[cur_policy_id]
        replay_buffer = replay_buffer_dict[cur_policy_id]

        # sample action and step
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, terminate, truncated, info = env.step(action)
        done = terminate or truncated

        # get replay buffer done
        if 'stage_done' in info:
            record_done = terminate or info['stage_done']
        else:
            record_done = terminate
        # record
        if record_done:
            success_task += 1
        episode_rewards.append(reward)

        # get mask and insert
        if not terminate or truncated:
            mask = 1.0
        else:
            mask = 0.0
        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=record_done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        # reset
        if done:
            # observation, done = env.reset(), False
            observation, _ = env.reset()
            done = False
            wandb.log({f"training/success_task": success_task}, step=i + FLAGS.pretrain_steps)
            wandb.log({f"training/episode_return": np.mean(episode_rewards)}, step=i + FLAGS.pretrain_steps)
            success_task = 0
            episode_rewards = []

        # do train
        if i >= FLAGS.start_training:
            for policy_id in policy_ids:
                # check valid
                if not update_all and policy_id != cur_policy_id:
                    continue
                if len(replay_buffer_dict[policy_id]) < FLAGS.start_training:
                    continue

                # do train
                train_replay_buffer = replay_buffer_dict[policy_id]
                train_agent = policy_dict[policy_id]
                train_demo_dataset = demo_dataset_dict[policy_id]

                online_batch = train_replay_buffer.sample(
                    int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
                )
                offline_batch = train_demo_dataset.sample(
                    int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
                )

                batch = combine(offline_batch, online_batch)

                train_agent, update_info = train_agent.update(batch, FLAGS.utd_ratio)
                policy_dict[policy_id] = train_agent

                if i % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        try:
                            wandb.log({f"training/{policy_id}_{k}": v.item()}, step=i + FLAGS.pretrain_steps)
                        except:
                            wandb.log({f"training/{policy_id}_{k}": v}, step=i + FLAGS.pretrain_steps)
        # do evaluate
        if i % FLAGS.eval_interval == 0:
        # if True:
            eval_info = evaluate_multi(
                policy_dict, 
                eval_env, 
                num_episodes=FLAGS.eval_episodes, 
                node_policy_dict=node_policy_dict,
                save_video=FLAGS.save_video
            )

            for k, v in eval_info.items():
                try:
                    wandb.log({f"evaluation/{k}": v.item()}, step=i + FLAGS.pretrain_steps)
                except:
                    wandb.log({f"evaluation/{k}": v}, step=i + FLAGS.pretrain_steps)

            if FLAGS.checkpoint_model and store_path is not None:
                for policy_id in policy_ids:
                    chkpt_dir = os.path.join(store_path, "policy_{}")
                    os.makedirs(chkpt_dir, exist_ok=True)
                    try:
                        checkpoints.save_checkpoint(
                            chkpt_dir, policy_dict[policy_id], step=i, keep=20, overwrite=True
                        )
                    except:
                        print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer and store_path is not None:
                for policy_id in policy_ids:
                    buffer_path = os.path.join(store_path, "replay_buffer_{}.pkl")
                    try:
                        with open(buffer_path, "wb") as f:
                            pickle.dump(replay_buffer_dict[policy_id], f, pickle.HIGHEST_PROTOCOL)
                    except:
                        print("Could not save agent buffer.")


def learn_policy(env, eval_env, demo_dataset, store_path=None, project_name=None):
    # check config
    FLAGS = TrainConfigSingle
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0
    flag_var = vars(FLAGS)

    # init wandb
    if project_name is None:
        project_name = FLAGS.project_name
    # wandb.init(project=project_name)
    # wandb.config.update({k:flag_var[k] for k in flag_var if '__' not in k})

    # init
    kwargs = {k:flag_var[k] for k in flag_var if '__' not in k}
    model_cls = kwargs.pop("model_cls")
    
    # init for record
    success_tasks = []
    episode_rewards = []
    observation_space, action_space = to_gym_space(env.observation_space, env.action_space)
    agent = globals()[model_cls].create(
        FLAGS.seed, observation_space, action_space, actor_lr=FLAGS.actor_lr, critic_lr=FLAGS.critic_lr,
        hidden_dims=FLAGS.hidden_dims, discount=FLAGS.discount, num_qs=FLAGS.num_qs, num_min_qs=FLAGS.num_min_qs, tau=FLAGS.tau, 
        critic_layer_norm=FLAGS.critic_layer_norm, temp_lr=FLAGS.temp_lr, init_temperature=FLAGS.init_temperature,
        target_entropy=FLAGS.target_entropy, backup_entropy=FLAGS.backup_entropy, critic_weight_decay=FLAGS.critic_weight_decay
    )

    replay_buffer = ReplayBuffer(
        observation_space, action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    for i in tqdm.tqdm(
        range(0, FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        offline_batch = demo_dataset.sample(FLAGS.batch_size * FLAGS.utd_ratio)
        batch = {}
        for k, v in offline_batch.items():
            batch[k] = v

        agent, update_info = agent.update(batch, FLAGS.utd_ratio)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"offline-evaluation/{k}": v}, step=i)

    observation, _ = env.reset()
    done = False
    best_success = 0
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, terminate, truncated, info = env.step(action)
        done = terminate or truncated

        # get replay buffer done
        record_done = terminate
        # record
        if record_done:
            success_tasks.append(1)
        episode_rewards.append(reward)

        if not terminate or truncated:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, _ = env.reset()
            done = False
            wandb.log({f"training/success_task": np.mean(success_tasks)}, step=i + FLAGS.pretrain_steps)
            wandb.log({f"training/episode_return": np.mean(episode_rewards)}, step=i + FLAGS.pretrain_steps)
            episode_rewards = []

        if i >= FLAGS.start_training:
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            )
            offline_batch = demo_dataset.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
            )

            batch = combine(offline_batch, online_batch)

            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    try:
                        wandb.log({f"training/{k}": v.item()}, step=i + FLAGS.pretrain_steps)
                    except:
                        wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)

        if i % FLAGS.eval_interval == 0:
            # do evaluation
            eval_info = evaluate_single(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )
            if eval_info['success rate'] > best_success:
                if store_path is not None:
                    chkpt_dir = os.path.join(store_path, "best_model")
                    checkpoints.save_checkpoint(
                        chkpt_dir, agent, step=i, keep=1, overwrite=True
                    )
            # log
            for k, v in eval_info.items():
                try:
                    wandb.log({f"evaluation/{k}": v.item()}, step=i + FLAGS.pretrain_steps)
                except:
                    wandb.log({f"evaluation/{k}": v}, step=i + FLAGS.pretrain_steps)
    # store
    if FLAGS.checkpoint_model and store_path is not None:
        chkpt_dir = os.path.join(store_path, "policy")
        checkpoints.save_checkpoint(
            chkpt_dir, agent, step=i, keep=1, overwrite=True
        )

    return agent


def load_policy(policy_path, env):
    # check config
    FLAGS = TrainConfigSingle
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0
    flag_var = vars(FLAGS)

    # init
    kwargs = {k:flag_var[k] for k in flag_var if '__' not in k}
    model_cls = kwargs.pop("model_cls")
    
    # init for record
    observation_space, action_space = to_gym_space(env.observation_space, env.action_space)
    agent = globals()[model_cls].create(
        FLAGS.seed, observation_space, action_space, actor_lr=FLAGS.actor_lr, critic_lr=FLAGS.critic_lr,
        hidden_dims=FLAGS.hidden_dims, discount=FLAGS.discount, num_qs=FLAGS.num_qs, num_min_qs=FLAGS.num_min_qs, tau=FLAGS.tau, 
        critic_layer_norm=FLAGS.critic_layer_norm, temp_lr=FLAGS.temp_lr, init_temperature=FLAGS.init_temperature,
        target_entropy=FLAGS.target_entropy, backup_entropy=FLAGS.backup_entropy, critic_weight_decay=FLAGS.critic_weight_decay
    )

    return checkpoints.restore_checkpoint(policy_path, agent)