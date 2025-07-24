today=`date +%T`
echo $today
gpu_id=$1
seed=1
algorithm=ours_dct
model=dpot_tiny
# 
dataset=sw2d_pda
CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
    -cf ./config/${algorithm}/${model}/dpot_${dataset}_${seed}.yaml
