export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
for HEAD_LR in 1e-2; do
    for BACKBONE_LR in 1e-2; do
        CUDA_VISIBLE_DEVICES=3 python main.py \
            --model-name-or-path google/vit-base-patch16-224-in21k \
            --dataset-name cars \
            --mode neat \
            --num_epochs 10 \
            --n_trial 1 \
            --head_lr $HEAD_LR \
            --weight_decay 4e-5 \
            --backbone_lr $BACKBONE_LR \
            --mhsa_dim 7  \
            --neat_mode 1
    done
done

for HEAD_LR in 5e-3; do
    for BACKBONE_LR in 5e-3; do
        CUDA_VISIBLE_DEVICES=3 python main.py \
            --model-name-or-path google/vit-base-patch16-224-in21k \
            --dataset-name cifar10 \
            --mode neat \
            --num_epochs 10 \
            --n_trial 1 \
            --head_lr $HEAD_LR \
            --weight_decay 9e-5 \
            --backbone_lr $BACKBONE_LR \
            --mhsa_dim 7 \
            --neat_mode 1
    done
done

for HEAD_LR in 5e-3; do
    for BACKBONE_LR in 5e-3; do
        CUDA_VISIBLE_DEVICES=3 python main.py \
            --model-name-or-path google/vit-base-patch16-224-in21k \
            --dataset-name cifar100 \
            --mode neat \
            --n_frequency 3000 \
            --num_epochs 10 \
            --n_trial 1 \
            --head_lr $HEAD_LR \
            --weight_decay 1e-4 \
            --backbone_lr $BACKBONE_LR \
            --mhsa_dim 7 \
            --neat_mode 1
    done
done

for HEAD_LR in 5e-3; do
    for BACKBONE_LR in 5e-3; do
        CUDA_VISIBLE_DEVICES=3 python main.py \
            --model-name-or-path google/vit-base-patch16-224-in21k \
            --dataset-name eurosat \
            --mode neat \
            --n_frequency 3000 \
            --num_epochs 10 \
            --n_trial 1 \
            --head_lr $HEAD_LR \
            --weight_decay 3e-4 \
            --backbone_lr $BACKBONE_LR \
            --mhsa_dim 7 \
            --neat_mode 1
    done
done

for HEAD_LR in 1e-2; do
    for BACKBONE_LR in 1e-2; do
        CUDA_VISIBLE_DEVICES=3 python main.py \
            --model-name-or-path google/vit-base-patch16-224-in21k \
            --dataset-name fgvc \
            --mode neat \
            --n_frequency 3000 \
            --num_epochs 10 \
            --n_trial 1 \
            --head_lr $HEAD_LR \
            --weight_decay 7e-5 \
            --backbone_lr $BACKBONE_LR \
            --mhsa_dim 7 \
            --neat_mode 1
    done
done


for HEAD_LR in 5e-3; do
    for BACKBONE_LR in 5e-3; do
        CUDA_VISIBLE_DEVICES=3 python main.py \
            --model-name-or-path google/vit-base-patch16-224-in21k \
            --dataset-name pets \
            --mode neat \
            --n_frequency 3000 \
            --num_epochs 10 \
            --n_trial 1 \
            --head_lr $HEAD_LR \
            --weight_decay 8e-4 \
            --backbone_lr $BACKBONE_LR \
            --mhsa_dim 7 \
            --neat_mode 1
    done
done


for HEAD_LR in 1e-2; do
    for BACKBONE_LR in 5e-3; do
        CUDA_VISIBLE_DEVICES=3 python main.py \
            --model-name-or-path google/vit-base-patch16-224-in21k \
            --dataset-name resisc \
            --mode neat \
            --n_frequency 3000 \
            --num_epochs 10 \
            --n_trial 1 \
            --head_lr $HEAD_LR \
            --weight_decay 3e-4 \
            --backbone_lr $BACKBONE_LR \
            --mhsa_dim 7 \
            --neat_mode 1
    done
done