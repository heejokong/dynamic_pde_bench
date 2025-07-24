today=`date +%T`
echo $today
gpu_id=$1
seed=1
algorithm=ours_dct_noise
model=fno
# 
dataset=ns2d_pda
CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
    -cf ./config/${algorithm}/${model}/fno_${dataset}_${seed}.yaml
