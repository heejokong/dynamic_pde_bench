today=`date +%T`
echo $today
gpu_id=$1
seed=1
algorithm=autoregression
model=dpot_tiny
# 
dataset=ns2d_pdb_M1
CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
    -cf ./config/${algorithm}/${model}/dpot_${dataset}_${seed}.yaml
