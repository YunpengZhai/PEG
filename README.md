# PEG

Official PyTorch implementation of Population-based Evolutionary Gaming. 

Zhai, Y., Peng, P., Jia, M. et al. Population-Based Evolutionary Gaming for Unsupervised Person Re-identification. Int J Comput Vis 131, 1–25 (2023). https://doi.org/10.1007/s11263-022-01693-7


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

To train person re-ID in paper, run this command:
```train
bash train_population.sh imagenet market1501 500
```

## In Details
```
├──  peg
│    └── population.py  - here's the operations of populations, including reproduction, mutation and selection.
│    └── trainer.py - here's the population mutual learning.
