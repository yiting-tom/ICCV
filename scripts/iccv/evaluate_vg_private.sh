#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6039
export CUDA_VISIBLE_DEVICES=3
export GPUS_PER_NODE=1

tag=vqa-P2_dif-0
########################## Evaluate VG ##########################
root=/home/P76104419/ICCV
user_dir=${root}/ofa_module
bpe_dir=${root}/utils/BPE
selected_cols=0,4,2,3

for prompt in '1' '2' '3' '4' '5'; do
    data=${root}/dataset/base64/vg/test_private-P${prompt}.csv
    split="private-P${prompt}" # vg_predict.json

    for ckpt in '8_9000' '10_11500' '7_8000' '5_5500' '5_6000' '3_3500' '7_7500' '4_4000' '5_5000' '6_7000' '10_12000' '1_1000' '9_10000' '6_6500' '2_2000' '3_3000' '4_4500' '2_1500' '8_9500' '1_500' '7_8500' '3_2500' '9_10500' '10_11000'; do
        path=/home/P76104419/ICCV/iccv_checkpoints/${tag}/checkpoint_${ckpt}.pt
        result_path=${root}/results/vg/${tag}/ckpt_${ckpt}
        echo "Evaluating ${path} and saving results to ${result_path} ..."

        # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 ../../evaluate.py \
            ${data} \
            --path=${path} \
            --user-dir=${user_dir} \
            --task=refcoco \
            --batch-size=20 \
            --log-format=simple --log-interval=10 \
            --seed=7 \
            --gen-subset=${split} \
            --results-path=${result_path} \
            --beam=12 \
            --min-len=4 \
            --max-len-a=0 \
            --max-len-b=4 \
            --no-repeat-ngram-size=3 \
            --fp16 \
            --num-workers=0 \
            --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
    done
done