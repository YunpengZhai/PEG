# PEG

Official PyTorch implementation of Population-based Evolutionary Game.

<p align="center">
    <img src=./img/framework.jpg width="800">
</p>

## 1. Requirements
### Environments
Currently, requires following packages
- python 3.6+
- torch 1.4+
- torchvision 0.5+
- CUDA 10.1+
- scikit-learn 0.22+
- faiss-gpu

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
### Pretrain
You can load the weights pretrained on imagenet, or download the [weights](https://pan.baidu.com/s/12rRW4cnMbxe6x1YxXqNbyw) pretrained on source data. (iinq)

You can also pretrain models by yourself.

### Person re-ID
To train person re-ID in paper, run this command:
```train
bash train_population.sh dukemtmc market1501 500
```
### Image retrieval
To train Stardford online products in paper, run this command:
```train
bash train_population_sop.sh imagenet sop 10000
```
## In Details
```
├──  peg
│    └── population.py  - here's the operations of populations, including reproduction, mutation and selection.
│    └── trainer.py - here's the mutual learning among populations.
