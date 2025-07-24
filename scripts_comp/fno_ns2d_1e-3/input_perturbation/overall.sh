today=`date +%T`
echo $today
gpu_id=$1
seed=1
model=fno
dataset=ns2d_1e-3
# 
# for baseline in 0_standard_teacher_forcing 1_scheduled_sampling_decode 1_scheduled_sampling_train 2_curriculum_learning 3_gaussian_noise_low 3_gaussian_noise_high
# for baseline in 0_standard_teacher_forcing 1_scheduled_sampling_train 1_scheduled_sampling_decode 2_curriculum_learning 3_gaussian_noise_low 3_gaussian_noise_high
for baseline in 0_standard_teacher_forcing
do
CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
    -cf ./config_comp/${model}_${dataset}/input_perturbation/${baseline}.yaml
done
