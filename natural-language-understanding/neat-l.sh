for scale in 0.1
do
    echo "Running with scale = $scale"
    CUDA_VISIBLE_DEVICES=2 python glue.py \
        --model_name_or_path roberta-base \
        --dataset cola \
        --task cola \
        --max_length 512 \
        --head_lr 1e-3 \
        --backbone_lr 1e-3 \
        --num_epoch 100 \
        --bs 64  \
        --dim 8 \
        --scale $scale \
        --seed 0 
done

for scale in 0.01
do
    echo "Running with scale = $scale"
    CUDA_VISIBLE_DEVICES=3 python glue.py \
        --model_name_or_path roberta-base \
        --dataset mrpc \
        --task mrpc \
        --max_length 512 \
        --head_lr 5e-3 \
        --backbone_lr 5e-3 \
        --num_epoch 100 \
        --bs 64  \
        --dim 8 \
        --scale $scale \
        --seed 0 
done


for scale in 0.01
do
    echo "Running with scale = $scale"
    CUDA_VISIBLE_DEVICES=2 python glue.py \
        --model_name_or_path roberta-base \
        --dataset qnli \
        --task qnli \
        --max_length 512 \
        --head_lr 1e-3 \
        --backbone_lr 1e-3 \
        --num_epoch 40 \
        --bs 32  \
        --scale $scale \
        --seed 0 \
        --dim 8 
done


for scale in 0.01
do
    echo "Running with scale = $scale"
    CUDA_VISIBLE_DEVICES=3 python glue.py \
        --model_name_or_path roberta-base \
        --dataset mnli \
        --task mnli \
        --max_length 512 \
        --head_lr 5e-3 \
        --backbone_lr 5e-3 \
        --num_epoch 20 \
        --bs 32  \
        --scale $scale \
        --seed 0 \
        --dim 8 
done

for scale in 0.01
do
    echo "Running with scale = $scale"
    CUDA_VISIBLE_DEVICES=3 python glue.py \
        --model_name_or_path roberta-base \
        --dataset qqp \
        --task qqp \
        --max_length 512 \
        --head_lr 5e-3 \
        --backbone_lr 5e-3 \
        --num_epoch 20 \
        --bs 64  \
        --scale $scale \
        --seed 0 \
        --dim 8 
done

for scale in 0.01
do
    echo "Running with scale = $scale"
    CUDA_VISIBLE_DEVICES=3 python glue.py \
        --model_name_or_path roberta-base \
        --dataset rte \
        --task rte \
        --max_length 512 \
        --head_lr 5e-3 \
        --backbone_lr 5e-3 \
        --num_epoch 100 \
        --bs 32  \
        --scale $scale \
        --seed 0 \
        --dim 8 
done



for scale in 0.01
do
    echo "Running with scale = $scale"
    CUDA_VISIBLE_DEVICES=3 python glue.py \
        --model_name_or_path roberta-base \
        --dataset sst2 \
        --task sst2 \
        --max_length 512 \
        --head_lr 5e-3 \
        --backbone_lr 5e-3 \
        --num_epoch 40 \
        --bs 32  \
        --scale $scale \
        --seed 0 \
        --dim 8 
done



for scale in 0.1
do
    echo "Running with scale = $scale"
    CUDA_VISIBLE_DEVICES=2 python glue.py \
        --model_name_or_path roberta-base \
        --dataset stsb \
        --task stsb \
        --max_length 512 \
        --head_lr 5e-3 \
        --backbone_lr 5e-3 \
        --num_epoch 100 \
        --bs 64  \
        --scale $scale \
        --seed 0 \
        --dim 8 
done

