# 1: feature 12
# 2: use Bayesian learning. 0 False 1 True.
# 3: G-metric hyper. empirically set to 5.
# 4: dataset. yewu_v3f or yewu_v3bL1.

for ((i=1;i<=20;i++))
do
    python rl_bid_agent.py \
    --folder 00CBRL4_${4}_eplearn_pb${5}_${1}_h${2}_GS${3}_sar_rnom0_sync1 \
    --ablation ${1} --use_history ${2} --history_type sar --norm_reward 0 --synchronous 1 --data_ver ${4} \
    --rev_scaler 1 --penalty 1 --agent sac --reward 6 --powerlaw_b 0.2 \
    --fc1_hidden 40 --tau 0.1 --buffer_size 5000 --max_loss 30. --gamma ${3} \
    --slot 30 --buffer 4 --replay 0 --nepoch 15 --gamma_nepoch 1 --force --learn_limits 1
done
