CUDA_VISIBLE_DEVICES=0 python main_rl.py --algo ppo --use-gae \
    --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 125 --num-mini-batch 8 \
    --log-interval 100 --eval-interval 100 \
    --use-linear-lr-decay --entropy-coef 0.01 \
    --approach ewc --experiment atari --ewc-lambda 5000 --seed 1 --ewc-online True
