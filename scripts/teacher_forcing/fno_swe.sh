today=`date +%T`
echo $today
gpu_id=$1
seed=1
algorithm=teacher_forcing
model=fno
# 
dataset=sw2d_pda
CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
    -cf ./config/${algorithm}/${model}/fno_${dataset}_${seed}.yaml
