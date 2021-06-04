#!/bin/sh
SOURCE=$1
TARGET=$2
CLUSTER=$3

python3 py/train.py  -ds ${SOURCE} -dt ${TARGET} --num-clusters ${CLUSTER} -b 64 --epochs 15 --iters 1000 --seed 2 --cross-iters 500 --genaration 3 --cluster dbscan \
	--seed 2 --mutation-r 0.5 \
	--logs-dir logs/${TARGET}/peg_2dgx_bs64 \
	--init-dir logs/${SOURCE}Pretrain