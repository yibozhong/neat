# Image Classification
## Environment setup

```bash
conda create -n neat-cv python=3.12
conda activate neat-cv
pip install evaluate
pip install peft
pip install transformer
```

## Fine-tune ViT on several tasks using vanilla Neat

There're 2 modes in neat to choose:

```bash
neat mode:
        1. neat on mhsa q, v (a.k.a. qv)
		2. neat on both mhsa q, v and mlp (a.k.a. qvmlp)

```

Run all the experiments using:

```bash
bash run.sh
```

## Stack up the layers

It is easy to stack up multiple intermediate layers with non-linear activation within to **further boost** the adaptation capability of Neat. Just add an argument `--multilayer` in the script and specify the depth (A, B plus intermediate layers). An example is:

```bash
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
            --mhsa_dim 7  \
            --neat_mode 1 \
            --multilayer \
            --depth 6 
    done
done
```

## Acknowledgments

The code is based on [fourierft](https://github.com/Chaos96/fourierft/tree/f8ab847bd7e7cb2f6a469bc5c8577fe96e5362bd/experiments/GLUE).