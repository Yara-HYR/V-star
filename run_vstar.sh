#!/bin/bash
# V-STAR Training: VED + Sibling-GRPO
# - freeze_lm=False: backbone is trainable
# - value_head_weight=1.0: joint value head training
# - use_ved=True: enable Value-Guided Efficient Decoding (V-STAR Sec 4.2)
# - use_sibling_grpo=True: enable Sibling-GRPO loss (V-STAR Sec 4.3)

export NCCL_IB_DISABLE=1

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
        --config_file ./config/zero2_opt.yaml \
        --num_processes 8 --main_process_port 29503 \
        rl.py \
        --model_path ./output/sft_qwen1.5b_industrial/final_checkpoint \
        --train_batch_size 64 \
        --eval_batch_size 128 \
        --num_train_epochs 2 \
        --gradient_accumulation_steps 2 \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --info_file ${info_file} \
        --category ${category} \
        --reward_type ranking \
        --num_generations 16 \
        --beam_search False \
        --test_during_training False \
        --temperature 1.0 \
        --learning_rate 1e-5 \
        --beta 1e-3 \
        --freeze_lm False \
        --value_head_weight 1.0 \
        --value_td_gamma 0.99 \
        --value_index_path ./data/Amazon/index/${category}.index.json \
        --value_emb_path ./data/Amazon/index/${category}.emb-qwen-td.npy \
        --use_ved True \
        --ved_budget_multiplier 1.0 \
        --ved_init_beam 8 \
        --ved_lambda 0.1 \
        --ved_beta_ucb 1.0 \
        --use_sibling_grpo True \
        --sibling_loss_weight 1.0 \
        --output_dir ./output/vstar_${category} \
        --sid_index_path ./data/Amazon/index/${category}.index.json \
        --item_meta_path ./data/Amazon/index/${category}.item.json
done
