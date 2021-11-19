#!/bin/sh
SOURCE=$1
TARGET=$2
CLUSTER=$3

python3 py/train.py  -ds ${SOURCE} -dt ${TARGET} --num-clusters ${CLUSTER} -b 256 --num-instances 16 --epochs 15 --iters 1000 --seed 2 --cross-iters 200 --genaration 3 --cluster dbscan \
	--seed 2 --mutation-r 0.5 \
	--logs-dir logs/${TARGET}/peg_bs256 \
