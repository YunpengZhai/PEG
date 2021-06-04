#!/bin/sh
SOURCE=$1
TARGET=$2
CLUSTER=$3

python3 py/train_sop.py -dt ${TARGET} --num-clusters ${CLUSTER} -b 64 --epochs 15 --iters 1500 --lr 0.0001 --crs-lr 0.0001 --mutation-r 0.5 --slots 2 \
	--cross-iters 800 \
	--logs-dir logs/${SOURCE}TO${TARGET}/train_population_bs64_sop \
