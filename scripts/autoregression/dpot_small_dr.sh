today=`date +%T`
echo $today
gpu_id=$1
seed=1
algorithm=autoregression
model=dpot_small
# 
dataset=dr_pdb
CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
    -cf ./config/${algorithm}/${model}/dpot_${dataset}_${seed}.yaml
