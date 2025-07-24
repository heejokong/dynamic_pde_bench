today=`date +%T`
echo $today
gpu_id=$1
seed=1
model=fno
dataset=ns2d_1e-5
# 
for baseline in 0_standard_mtl 1_mtl_pcgrad 1_mtl_cagrad 2_adversarial_training 3_ours_inter_objective 4_ours_intra_objective 5_ours_composite
do
CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
    -cf ./config_comp/${model}_${dataset}/multi_objective/${baseline}.yaml
done
