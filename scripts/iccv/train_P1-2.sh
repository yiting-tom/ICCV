#!/usr/bin/env
export MASTER_PORT=6052
root=/home/P76104419/ICCV
log_dir=${root}/logs/iccv
save_dir=${root}/iccv_checkpoints
mkdir -p $log_dir $save_dir
bpe_dir=${root}/utils/BPE
user_dir=${root}/ofa_module
DEVICES=0,1
NODE_NUM=2

# # ========================================= P1 =========================================
# prompt=1
# tag=vqa-P${prompt}_dif-0
# tensorboard_dir=${root}/tensorboard/iccv/${tag}

# data_dir=${root}/dataset/base64/vg
# data=${data_dir}/train-P${prompt}.csv,${data_dir}/test_public.csv

# restore_file=${root}/backbone/ofa_large.pt
# selected_cols=0,4,2,3

# task=refcoco
# arch=ofa_large
# criterion=adjust_label_smoothed_cross_entropy
# label_smoothing=0.1
# lr=3e-5
# max_epoch=10
# warmup_ratio=0.06
# batch_size=4
# update_freq=8
# resnet_drop_path_rate=0.0
# encoder_drop_path_rate=0.2
# decoder_drop_path_rate=0.2
# dropout=0.1
# attention_dropout=0.0
# max_src_length=80
# max_tgt_length=20
# num_bins=1000
# patch_image_size=512

# log_file=${log_dir}/${tag}.log
# save_path=${save_dir}/${tag}
# mkdir -p $save_path

# CUDA_VISIBLE_DEVICES=${DEVICES} python3 -m torch.distributed.launch --nproc_per_node=${NODE_NUM} --master_port=${MASTER_PORT} ${root}/train.py \
#   $data \
#   --selected-cols=${selected_cols} \
#   --bpe-dir=${bpe_dir} \
#   --user-dir=${user_dir} \
#   --restore-file=${restore_file} \
#   --reset-optimizer --reset-dataloader --reset-meters \
#   --save-dir=${save_path} \
#   --task=${task} \
#   --arch=${arch} \
#   --criterion=${criterion} \
#   --label-smoothing=${label_smoothing} \
#   --batch-size=${batch_size} \
#   --update-freq=${update_freq} \
#   --encoder-normalize-before \
#   --decoder-normalize-before \
#   --share-decoder-input-output-embed \
#   --share-all-embeddings \
#   --layernorm-embedding \
#   --patch-layernorm-embedding \
#   --code-layernorm-embedding \
#   --resnet-drop-path-rate=${resnet_drop_path_rate} \
#   --encoder-drop-path-rate=${encoder_drop_path_rate} \
#   --decoder-drop-path-rate=${decoder_drop_path_rate} \
#   --dropout=${dropout} \
#   --attention-dropout=${attention_dropout} \
#   --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
#   --lr-scheduler=polynomial_decay --lr=${lr} \
#   --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
#   --log-format=simple --log-interval=10 \
#   --fixed-validation-seed=7 \
#   --no-epoch-checkpoints --keep-best-checkpoints=1 \
#   --save-interval=1 --validate-interval=1 \
#   --save-interval-updates=500 --validate-interval-updates=500 \
#   --eval-acc \
#   --eval-args='{"beam":5,"min_len":4,"max_len_a":0,"max_len_b":4}' \
#   --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
#   --max-src-length=${max_src_length} \
#   --max-tgt-length=${max_tgt_length} \
#   --find-unused-parameters \
#   --add-type-embedding \
#   --scale-attn \
#   --scale-fc \
#   --scale-heads \
#   --disable-entangle \
#   --num-bins=${num_bins} \
#   --patch-image-size=${patch_image_size} \
#   --fp16 \
#   --fp16-scale-window=512 \
#   --tensorboard-logdir=${tensorboard_dir} \
#   --num-workers=0 >${log_file} 2>&1
# # ========================================= P1 =========================================

# ========================================= P2 =========================================
prompt=2
tag=vqa-P${prompt}_dif-0
tensorboard_dir=${root}/tensorboard/iccv/${tag}

data_dir=${root}/dataset/base64/vg
data=${data_dir}/train-P${prompt}.csv,${data_dir}/test_public-P${prompt}.csv

restore_file=${root}/backbone/ofa_large.pt
selected_cols=0,4,2,3

task=refcoco
arch=ofa_large
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=3e-5
max_epoch=10
warmup_ratio=0.06
batch_size=4
update_freq=8
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=20
num_bins=1000
patch_image_size=512

log_file=${log_dir}/${tag}.log
save_path=${save_dir}/${tag}
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=${DEVICES} python3 -m torch.distributed.launch --nproc_per_node=${NODE_NUM} --master_port=${MASTER_PORT} ${root}/train.py \
  $data \
  --selected-cols=${selected_cols} \
  --bpe-dir=${bpe_dir} \
  --user-dir=${user_dir} \
  --restore-file=${restore_file} \
  --reset-optimizer --reset-dataloader --reset-meters \
  --save-dir=${save_path} \
  --task=${task} \
  --arch=${arch} \
  --criterion=${criterion} \
  --label-smoothing=${label_smoothing} \
  --batch-size=${batch_size} \
  --update-freq=${update_freq} \
  --encoder-normalize-before \
  --decoder-normalize-before \
  --share-decoder-input-output-embed \
  --share-all-embeddings \
  --layernorm-embedding \
  --patch-layernorm-embedding \
  --code-layernorm-embedding \
  --resnet-drop-path-rate=${resnet_drop_path_rate} \
  --encoder-drop-path-rate=${encoder_drop_path_rate} \
  --decoder-drop-path-rate=${decoder_drop_path_rate} \
  --dropout=${dropout} \
  --attention-dropout=${attention_dropout} \
  --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
  --lr-scheduler=polynomial_decay --lr=${lr} \
  --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
  --log-format=simple --log-interval=10 \
  --fixed-validation-seed=7 \
  --no-epoch-checkpoints --keep-best-checkpoints=1 \
  --save-interval=1 --validate-interval=1 \
  --save-interval-updates=500 --validate-interval-updates=500 \
  --eval-acc \
  --eval-args='{"beam":5,"min_len":4,"max_len_a":0,"max_len_b":4}' \
  --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
  --max-src-length=${max_src_length} \
  --max-tgt-length=${max_tgt_length} \
  --find-unused-parameters \
  --add-type-embedding \
  --scale-attn \
  --scale-fc \
  --scale-heads \
  --disable-entangle \
  --num-bins=${num_bins} \
  --patch-image-size=${patch_image_size} \
  --fp16 \
  --fp16-scale-window=512 \
  --tensorboard-logdir=${tensorboard_dir} \
  --num-workers=0 >${log_file} 2>&1
# ========================================= P2 =========================================

# ========================================= P5 =========================================
prompt=5
tag=vqa-P${prompt}_dif-0
tensorboard_dir=${root}/tensorboard/iccv/${tag}

data_dir=${root}/dataset/base64/vg
data=${data_dir}/train-P${prompt}.csv,${data_dir}/test_public-P${prompt}.csv

restore_file=${root}/backbone/ofa_large.pt
selected_cols=0,4,2,3

task=refcoco
arch=ofa_large
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=3e-5
max_epoch=10
warmup_ratio=0.06
batch_size=4
update_freq=8
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=20
num_bins=1000
patch_image_size=512

log_file=${log_dir}/${tag}.log
save_path=${save_dir}/${tag}
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=${DEVICES} python3 -m torch.distributed.launch --nproc_per_node=${NODE_NUM} --master_port=${MASTER_PORT} ${root}/train.py \
  $data \
  --selected-cols=${selected_cols} \
  --bpe-dir=${bpe_dir} \
  --user-dir=${user_dir} \
  --restore-file=${restore_file} \
  --reset-optimizer --reset-dataloader --reset-meters \
  --save-dir=${save_path} \
  --task=${task} \
  --arch=${arch} \
  --criterion=${criterion} \
  --label-smoothing=${label_smoothing} \
  --batch-size=${batch_size} \
  --update-freq=${update_freq} \
  --encoder-normalize-before \
  --decoder-normalize-before \
  --share-decoder-input-output-embed \
  --share-all-embeddings \
  --layernorm-embedding \
  --patch-layernorm-embedding \
  --code-layernorm-embedding \
  --resnet-drop-path-rate=${resnet_drop_path_rate} \
  --encoder-drop-path-rate=${encoder_drop_path_rate} \
  --decoder-drop-path-rate=${decoder_drop_path_rate} \
  --dropout=${dropout} \
  --attention-dropout=${attention_dropout} \
  --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
  --lr-scheduler=polynomial_decay --lr=${lr} \
  --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
  --log-format=simple --log-interval=10 \
  --fixed-validation-seed=7 \
  --no-epoch-checkpoints --keep-best-checkpoints=1 \
  --save-interval=1 --validate-interval=1 \
  --save-interval-updates=500 --validate-interval-updates=500 \
  --eval-acc \
  --eval-args='{"beam":5,"min_len":4,"max_len_a":0,"max_len_b":4}' \
  --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
  --max-src-length=${max_src_length} \
  --max-tgt-length=${max_tgt_length} \
  --find-unused-parameters \
  --add-type-embedding \
  --scale-attn \
  --scale-fc \
  --scale-heads \
  --disable-entangle \
  --num-bins=${num_bins} \
  --patch-image-size=${patch_image_size} \
  --fp16 \
  --fp16-scale-window=512 \
  --tensorboard-logdir=${tensorboard_dir} \
  --num-workers=0 >${log_file} 2>&1