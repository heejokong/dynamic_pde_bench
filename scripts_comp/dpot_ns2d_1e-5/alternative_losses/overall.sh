today=`date +%T`
echo $today
gpu_id=$1
seed=1
model=dpot
dataset=ns2d_1e-5
# 
# for baseline in 0_standard_autoregression 1_pushforward 2_sobolev_losses_low 2_sobolev_losses_high 3_dissipative_regularization 4_pde_refiner_low 4_pde_refiner_high
for baseline in 0_standard_autoregression 1_pushforward 2_sobolev_losses_low 2_sobolev_losses_high 3_dissipative_regularization
do
CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
    -cf ./config_comp/${model}_${dataset}/alternative_losses/${baseline}.yaml
done
