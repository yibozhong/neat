# Natural Language Understanding

## Environment setup

```bash
conda create -n neat-nlu python=3.12
conda activate neat-nlu
pip install -r requirements.txt
```

## Fine-tune

Use the scripts to finetune with neat:

```bash
bash neat-l.sh
bash neat-s.sh
```

The scripts already have the best configuration from our experiments. However, the hyperparameters can still be flexibly modified. The `neat-l.sh` script corresponds to settings for Neat-L, while `neat-s.sh` corresponds to settings for Neat-S in our experiments.

## Acknowledgments

The code is based on [fourierft](https://github.com/Chaos96/fourierft/tree/f8ab847bd7e7cb2f6a469bc5c8577fe96e5362bd/experiments/GLUE).