# PEG

Official PyTorch implementation of Population-based Evolutionary Game.

<!-- <p align="center">
    <img src=./img/ARPL.jpg width="800">
</p> -->

## 1. Requirements
### Environments
Currently, requires following packages
- python 3.6+
- torch 1.4+
- torchvision 0.5+
- CUDA 10.1+
- scikit-learn 0.22+

### Datasets
Download Market-1501 dataset to ./data

For example:

```
├──  data  
│    └── market1501  
│        └── Market-1501-v15.09.15
│            └── bounding_box_train
│            └── bounding_box_test
│            └── query
```

## 2. Training & Evaluation

### Person re-ID
To train person re-ID in paper, run this command:
```train
bash train_population.sh dukemtmc market1501 500
```
