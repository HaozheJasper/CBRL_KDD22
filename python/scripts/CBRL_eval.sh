# 1: feature 12
# 2: use Bayesian learning. 0 False 1 True.
# 3: dataset. yewu_v3f or yewu_v3bL1.
# 4: family folder for loading models
parentfolder=$5
for dir in ${parentfolder}/*
do
    echo $dir
    python rl_bid_agent.py \
    --test_only --resume --restore_dir ${dir} \
    --folder 00CBRLeval_${3}_ep1_pb02_${1}_h${2}_sar_rnom0_sync1 \
    --ablation ${1} --use_history ${2} --history_type sar --norm_reward 0 --synchronous 1 --data_ver ${3} \
    --rev_scaler 1 --penalty 1 --agent sac --reward 6 --powerlaw_b 0.2 \
    --fc1_hidden 40 --tau 0.1 --buffer_size 5000 --max_loss 30. \
    --slot 30 --buffer 4 --replay 0 --gamma_nepoch 1 --force --gamma 5
done

