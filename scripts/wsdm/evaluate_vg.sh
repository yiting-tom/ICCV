#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6939
export CUDA_VISIBLE_DEVICES=2
export GPUS_PER_NODE=1

########################## Evaluate VG ##########################
data=/wsdm/dataset/base64/test_private.csv
user_dir=/wsdm/ofa_module
bpe_dir=/wsdm/utils/BPE
selected_cols=0,4,2,3
split="vg" # vg_predict.json

# checkpoint_num_list="7_8000 7_7500 6_7000"
# checkpoint_num_list="6_6500 5_6000 5_5500"
checkpoint_num_list="4_5000 4_4500"
model="vqa-0_dif-0"

for checkpoint_num in ${checkpoint_num_list}; do
    path=/wsdm/iccv_checkpoints/${model}/checkpoint_${checkpoint_num}.pt
    result_path=/wsdm/results/vg/${model}/${checkpoint_num}
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
        --beam=15 \
        --min-len=4 \
        --max-len-a=0 \
        --max-len-b=4 \
        --no-repeat-ngram-size=3 \
        --fp16 \
        --num-workers=0 \
        --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
done