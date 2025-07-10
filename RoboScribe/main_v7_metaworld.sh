train_metaworld(){
    CUDA_VISIBLE_DEVICES=$1 python main_v7.py \
    --seed 0 \
    --log_path tree_metaworld_seed0.txt \
    --skill_path store/tree_metaworld_seed0/debug_skill \
    --demo_path store/demo_store/new_metaworld.npy \
    --policy_path store/tree_metaworld_seed0/debug_policy_{} \
    --fig_path store/tree_metaworld_seed0/states \
    --rl_timesteps 800000 \
    --rl_batch 512 \
    --eval_traj_len 2000 \
    --train_traj_len 2000 \
    --collect_epoch 20 \
    --collect_len -1 \
    --collect_traj_len 2000 \
    --final_goal_threshold 0.5 \
    --success_threshold 0.8 \
    --fail_success_threshold 0.8 \
    --extra_threshold 0.7 \
    --rew_threshold 0.5 \
    --reward_num 10 \
    --dso_hard_code \
    --total_iteration 50 \
    --hold_len 3 \
    --eval_epoch 100 \
    --final_eval_epoch 500 \
    --predicate_type 'tree' \
    --dense_type 'cont' \
    --n_cpu 8 \
    --obs_abs \
    --hold_rule \
    --env_name 'metaworld' \
    --shift_positive \
    --policy_type 'sac_lag' \
    --enforce_collect \
    --lagrangian_mode \
    --reward_refine \
    --use_best \
    --min_samples_mode 'pos_ratio' \
    --min_samples_leaf 0.5 \
    --prune_rule \
    --prog_sim
}


train_metaworld 2