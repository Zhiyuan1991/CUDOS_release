# Continual Unsupervised Disentangling of Self-Organizing Representations

This repository provides the implementation for the paper:

[Continual Unsupervised Disentangling of Self-Organizing Representations](https://openreview.net/forum?id=ih0uFRFhaZZ)

Published on ICLR 2023.


## Environment

* Python >= 3.8
* tensorflow>=2.5
* PyTorch >= 1.7
* matplotlib
* numpy
* scipy
* PIL

Or import from Conda environment "py38.yml".

## Examples
Data are supposed be in "../../Data". Or be defined in "data_manager.py".

Run training on continual 3DShape dataset:
```
python main_som.py -mode=1 -checkpoint_dir=checkpoints/3dshapes_cudos_r1 -gpu_usage=1. \
-epoch_size=50 -epoch_size_t2=15 -gamma=1 -start_task=1 -max_task=2 -replay_num=50000 -min_step=0 -max_step=3000 \
-z_dim=15 -task=2 -update_som="True" -sparse_coding="True" -Bayesian_SOM="True" -use_replay="True"
```
Run metrics on continual 3DShape dataset:
```
python main_som.py -mode=3 -checkpoint_dir=checkpoints/3dshapes_cudos_r1 -gpu_usage=.5 \
-z_dim=15 -task=2 -sparse_coding_MIG="True" -sparse_coding="True" -Bayesian_SOM="True"
```
