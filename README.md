# PEG

Official PyTorch implementation of Population-based Evolutionary Gaming. 
"Population-Based Evolutionary Gaming for Unsupervised Person Re-identification" IJCV 2023.

@article{cite-key,
	author = {Zhai, Yunpeng and Peng, Peixi and Jia, Mengxi and Li, Shiyong and Chen, Weiqiang and Gao, Xuesong and Tian, Yonghong},
	date = {2023/01/01},
	date-added = {2023-04-14 15:44:15 +0800},
	date-modified = {2023-04-14 15:44:15 +0800},
	doi = {10.1007/s11263-022-01693-7},
	id = {Zhai2023},
	isbn = {1573-1405},
	journal = {International Journal of Computer Vision},
	number = {1},
	pages = {1--25},
	title = {Population-Based Evolutionary Gaming for Unsupervised Person Re-identification},
	url = {https://doi.org/10.1007/s11263-022-01693-7},
	volume = {131},
	year = {2023},
	bdsk-url-1 = {https://doi.org/10.1007/s11263-022-01693-7}}


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
