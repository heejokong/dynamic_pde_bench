today=`date +%T`
echo $today
gpu_id=$1
seed=1
algorithm=ours_dct
model=fno
# 
dataset=dr_pdb
CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
    -cf ./config/${algorithm}/${model}/fno_${dataset}_${seed}.yaml
