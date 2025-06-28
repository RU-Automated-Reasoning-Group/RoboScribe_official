eval_multi_tower_tower5_iter_away(){
    CUDA_VISIBLE_DEVICES=$1 python3.9 main_v7_tower_transfer.py \
    --seed 0 \
    --log_path tree_tower5_transfer.txt \
    --skill_path store/tree_tower5_transfer/debug_skill \
    --demo_path store/demo_store/tower_5.npy \
    --policy_path store/tree_tower5_transfer/debug_policy_{} \
    --fig_path store/tree_tower5_transfer/states \
    --eval_store_path store/tree_tower5_transfer/eval_figs \
    --load_path store_backup/tree_tower5_transfer_load/debug_skill \
    --rl_timesteps 2000000 \
    --rl_batch 512 \
    --eval_traj_len 900 \
    --train_traj_len 900 \
    --collect_epoch 20 \
    --collect_len -1 \
    --collect_traj_len 250 \
    --final_goal_threshold 0.5 \
    --success_threshold 0.8 \
    --fail_success_threshold 0.8 \
    --extra_threshold 0.7 \
    --rew_threshold 0.5 \
    --reward_num 1 \
    --dso_hard_code \
    --total_iteration 50 \
    --hold_len 3 \
    --eval_epoch 100 \
    --final_eval_epoch 700 \
    --predicate_type 'tree' \
    --dense_type 'cont' \
    --n_cpu 8 \
    --obs_abs \
    --hold_rule \
    --shift_positive \
    --policy_type 'awac' \
    --enforce_collect \
    --lagrangian_mode \
    --reward_refine \
    --stage_reuse
}

eval_multi_tower_tower5_iter_away 0