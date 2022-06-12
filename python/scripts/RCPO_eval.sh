# 1: dataset. yewu_v3f or yewu_v3bL1.
# 2: family folder for loading models
parentfolder=$2
for dir in ${parentfolder}/*
do
    echo $dir
    python rl_bid_agent.py \
    --test_only --resume --restore_dir ${dir} \
    --folder Lagrangian4eval_${1}_mrand_t0 \
    --ablation 12 --use_history 0 --history_type sar --norm_reward 0 --synchronous 1 \
    --data_ver ${1} --init_multiplier 0 --tune_multiplier 0 \
    --rev_scaler 1 --penalty 1 --agent sac --reward 3 --discount 1 \
    --fc1_hidden 40 --tau 0.1 --buffer_size 5000 --max_loss 30. \
    --slot 30 --buffer 4 --replay 0 --nepoch 15 --gamma_nepoch 1 --force
done
