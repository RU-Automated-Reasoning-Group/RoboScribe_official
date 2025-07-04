train_drawer_pickplacecubemulti(){
    CUDA_VISIBLE_DEVICES=$1 python3.9 main_v7.py \
    --seed 0 \
    --log_path tree_drawer_pickplacecubemulti.txt \
    --skill_path store/tree_drawer_pickplacecubemulti/debug_skill \
    --demo_path store/demo_store/opendrawer_pickplacecube3_crop.npy \
    --preset_demo_path store/demo_store/opendrawer_pickplacecube2_crop.npy \
    --policy_path store/tree_drawer_pickplacecubemulti/debug_policy_{} \
    --fig_path store/tree_drawer_pickplacecubemulti/states \
    --rl_timesteps 1500000 \
    --rl_batch 512 \
    --eval_traj_len 200 \
    --train_traj_len 200 \
    --collect_epoch 20 \
    --collect_len -1 \
    --collect_traj_len 200 \
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
    --env_name 'drawer_pickplacecubemulti' \
    --shift_positive \
    --policy_type 'sac_lag' \
    --enforce_collect \
    --lagrangian_mode \
    --reward_refine \
    --use_best \
    --min_samples_mode 'min_ratio' \
    --min_samples_leaf 0.5 \
    --last_states \
    --prune_rule \
    --prog_sim \
    --stage_reuse \
    --set_block_num 3 \
    --preset_block_num 2
}


train_drawer_pickplacecubemulti 0
