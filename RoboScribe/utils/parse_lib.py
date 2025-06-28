import argparse

def get_parse():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--seed', type=int ,default=0, help='seed set for random')
    parser.add_argument('--log_path', type=str, default='log.txt', help='name of log file')
    parser.add_argument('--skill_path', type=str, default='store/skill.pth', help='path to store skill list')

    # demonstration
    parser.add_argument('--demo_path', type=str, default=None, help='path to store human demonstration')
    parser.add_argument('--preset_demo_path', type=str, default=None, help='path to store human demonstration')
    parser.add_argument('--img_path', type=str, default=None, help='path for demo images')
    parser.add_argument('--demo_sample_num', type=int, default=None, help='sample specific number of demo (for ablation study)')

    # environment
    parser.add_argument('--env_name', type=str, default='FetchPickAndPlace-v0', help='name of environment')
    parser.add_argument('--simple_obs', default=False, action='store_true', help='simple observation shape')
    parser.add_argument('--hold_len', type=int, default=1, help='trajectory length to hold success reward')
    parser.add_argument('--set_traj_limit', default=False, action='store_true', help='True if set trajectory length limit for each policy')
    parser.add_argument('--disable_rew_add', default=False, action='store_true', help='True if not plus 1 for reward')
    parser.add_argument('--set_block_num', type=int, default=None, help='set block number for observation transit if set')
    parser.add_argument('--preset_block_num', type=int, default=None, help='set smaller block number to get predicate graph')

    # policy
    parser.add_argument('--rl_batch', type=int, default=64, help='batch size for policy training')
    parser.add_argument('--rl_n_steps', type=int, default=2048, help='number of steps for policy training')
    parser.add_argument('--rl_timesteps', type=int, default=2e5, help='total timesteps for policy training')
    parser.add_argument('--finetune_rl_timesteps', type=int, default=2e5, help='total timesteps for policy finetuning')
    parser.add_argument('--eval_epoch', type=int, default=5, help='epoch for policy evaluation')
    parser.add_argument('--final_eval_epoch', type=int, default=50, help='epoch for policy evaluation on entire stage')
    parser.add_argument('--eval_coll_epoch', type=int, default=500, help='epoch for collect data for classifier')
    parser.add_argument('--eval_traj_len', type=int, default=100, help='max timestep for each trajectory')
    parser.add_argument('--pick_eval', default=False, action='store_true', help='True if only evaluating pick branch classifier')
    parser.add_argument('--train_traj_len', type=int, default=100, help='max timestep for each trajectory training')
    parser.add_argument('--policy_path', type=str, default='policy_{}', help='path to store policy learning results')
    parser.add_argument('--policy_type', type=str, default='ppo', choices=['ppo', 'sac', 'sac_lag', 'sac_lag_d', 'sac_lag_bag_d', 'sac_lag_rce', 'sac_rce', 'rce', 'awac', 'rlpd'], help='policy model')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of process to train policy')
    parser.add_argument('--traj_limit_mode', default=False, action='store_true', help='True if limit trajectory length based on experiment (recommend evaluation epoch no less than 100)')
    parser.add_argument('--d_dist', default=False, action='store_true', help='set if use distance regression for discriminator training')
    parser.add_argument('--use_best', default=False, action='store_true', help='set if use the best policy rather than the last one')

    # program
    parser.add_argument('--dso_hard_code', default=False, action='store_true', help='set if hard code to be true')
    parser.add_argument('--config_path', type=str, default=None, help='path of dso config')
    parser.add_argument('--only_old', default=False, action='store_true', help='set if only use original rules for predicate termination')
    parser.add_argument('--prog_sim', default=False, action='store_true', help='set if simplify decision tree by merging')
    parser.add_argument('--min_samples_leaf', type=float, default=None, help='minimum sample required to be covered by a node in decision tree')
    parser.add_argument('--min_samples_mode', type=str, choices=[None, 'num', 'ratio', 'pos_ratio', 'min_ratio'], default=None, help='minimum sample mode')
    parser.add_argument('--negative_state_crop', type=float, default=1.0, help='crop trajectory for negative states')
    parser.add_argument('--last_states', default=False, action='store_true', help='set if only use last states as negative state for stage generation')
    parser.add_argument('--first_states', default=False, action='store_true', help='set if only use first states as negative state for stage generation')

    # framework
    parser.add_argument('--final_goal_threshold', type=float, default=1.0, help='threshold for success rate overall')
    parser.add_argument('--success_threshold', type=float, default=1.0, help='threshold for policy success')
    parser.add_argument('--fail_success_threshold', type=float, default=1.0, help='threshold for policy succes in fail branch')
    parser.add_argument('--extra_threshold', type=float, default=None, help='set if generate fail branch')
    parser.add_argument('--rew_threshold', type=float, default=0.5, help='threshold for success of reward function')
    parser.add_argument('--collect_epoch', type=int, default=10, help='epoch to collect state')
    parser.add_argument('--collect_retrain_epoch', type=int, default=100, help='epoch to collect state')
    parser.add_argument('--collect_len', type=int, default=5, help='number of steps to collect')
    parser.add_argument('--collect_traj_len', type=int, default=20, help='number of steps to rollout')
    parser.add_argument('--collect_traj_freq', type=int, default=1, help='how many states for 1 store')
    parser.add_argument('--augment_success', default=False, action='store_true', help='True if augment positive states with success trajectories')
    parser.add_argument('--total_iteration', type=int, default=50, help='limited number of iteration for framework training')
    parser.add_argument('--predicate_type', type=str, choices=['dso', 'neural', 'tree', 'forest', 'rce'], default='dso', help='predicate learner type: [dso, neural]')
    parser.add_argument('--dense_type', type=str, choices=['cut', 'cont'], default='cut', help='dense type for reward function: [cut, cont]')
    parser.add_argument('--predicate_retrain', default=False, action='store_true', help='set if adjust predicate when fail')
    parser.add_argument('--shift_positive', default=False, action='store_true', help='set if shift threshold towards positive states')
    parser.add_argument('--hold_rule', default=False, action='store_true', help='True if covered rules need to be held')
    parser.add_argument('--split_predicate', default=False, action='store_true', help='True if split predicate based on "or"')
    parser.add_argument('--shift_alpha', type=float, default=1.0, help='alpha to keep shifited value: alpha*new_val + (1-alpha)*old_val')
    parser.add_argument('--lagrangian_mode', default=False, action='store_true', help='True if use lagrangian method')
    parser.add_argument('--disable_fail', default=False, action='store_true', help='True if disable fail search')
    parser.add_argument('--reward_refine', default=False, action='store_true', help='refine reward rather than create fail branch if specific')
    parser.add_argument('--non_const', default=False, action='store_true', help='Set if add negative constraint to goal predicate')
    parser.add_argument('--reward_num', type=int, default=1, help='number of candidate reward function considered')
    parser.add_argument('--stage_reuse', default=False, action='store_true', help='True if reuse previous stages')
    parser.add_argument('--prune_rule', default=False, action='store_true', help='True if prune rules with at least half number of positive states')

    # evaluation
    parser.add_argument('--eval_store_path', type=str, default='store/debug_eval', help='path to store evaluation results')

    # continue training
    parser.add_argument('--load_path', type=str, default=None, help='continue training with exist reward and policy')

    # debug
    parser.add_argument('--fig_path', type=str, default=None, help='path to store figures')
    parser.add_argument('--obs_abs', default=False, action='store_true', help='set if apply abstract state')
    parser.add_argument('--enforce_collect', default=False, action='store_true', help='set if enforce the number of episodes collected')
    parser.add_argument('--debug_mode', default=False, action='store_true')

    # for rce
    parser.add_argument('--n_steps', type=int, default=10)
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--log_std_min', type=float, default=-2.)
    parser.add_argument('--log_std_max', type=float, default=-20.)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy_coef', type=float, default=1e-4)
    parser.add_argument('--rollout_steps', type=int, default=201)
    parser.add_argument('--iterations', type=int, default=15000)
    parser.add_argument('--evaluation_rollouts', type=int, default=5)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--display_loss_interval', type=int, default=1000)
    parser.add_argument('--init_num_n_step_trajs', type=int, default=1000)
    parser.add_argument('--polyak', type=float, default=0.95)

    args = parser.parse_args()

    return args