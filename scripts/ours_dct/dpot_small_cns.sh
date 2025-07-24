today=`date +%T`
echo $today
gpu_id=$1
seed=1
algorithm=ours_dct
model=dpot_small
# 
dataset=ns2d_pdb_M1
CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
    -cf ./config/${algorithm}/${model}/dpot_${dataset}_${seed}.yaml
