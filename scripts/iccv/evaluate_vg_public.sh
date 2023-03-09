#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6039
export CUDA_VISIBLE_DEVICES=2
export GPUS_PER_NODE=1

tag=vqa-P2_dif-0

########################## Evaluate VG ##########################
prompt=1
root=/home/P76104419/ICCV
data=${root}/dataset/base64/vg/test_public-P${prompt}.csv
user_dir=${root}/ofa_module
bpe_dir=${root}/utils/BPE
selected_cols=0,4,2,3
split="public-P${prompt}" # vg_predict.json
path=/home/P76104419/ICCV/iccv_checkpoints/${tag}/checkpoint_best.pt
result_path=${root}/results/vg/${tag}
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


########################## Evaluate VG ##########################
prompt=2
root=/home/P76104419/ICCV
data=${root}/dataset/base64/vg/test_public-P${prompt}.csv
user_dir=${root}/ofa_module
bpe_dir=${root}/utils/BPE
selected_cols=0,4,2,3
split="public-P${prompt}" # Px_predict.json
path=/home/P76104419/ICCV/iccv_checkpoints/${tag}/checkpoint_best.pt
result_path=${root}/results/vg/${tag}
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

########################## Evaluate VG ##########################
prompt=3
root=/home/P76104419/ICCV
data=${root}/dataset/base64/vg/test_public-P${prompt}.csv
user_dir=${root}/ofa_module
bpe_dir=${root}/utils/BPE
selected_cols=0,4,2,3
split="public-P${prompt}" # Px_predict.json
path=/home/P76104419/ICCV/iccv_checkpoints/${tag}/checkpoint_best.pt
result_path=${root}/results/vg/${tag}
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

########################## Evaluate VG ##########################
prompt=4
root=/home/P76104419/ICCV
data=${root}/dataset/base64/vg/test_public-P${prompt}.csv
user_dir=${root}/ofa_module
bpe_dir=${root}/utils/BPE
selected_cols=0,4,2,3
split="public-P${prompt}" # Px_predict.json
path=/home/P76104419/ICCV/iccv_checkpoints/${tag}/checkpoint_best.pt
result_path=${root}/results/vg/${tag}
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

########################## Evaluate VG ##########################
prompt=5
root=/home/P76104419/ICCV
data=${root}/dataset/base64/vg/test_public-P${prompt}.csv
user_dir=${root}/ofa_module
bpe_dir=${root}/utils/BPE
selected_cols=0,4,2,3
split="public-P${prompt}" # Px_predict.json
path=/home/P76104419/ICCV/iccv_checkpoints/${tag}/checkpoint_best.pt
result_path=${root}/results/vg/${tag}
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
