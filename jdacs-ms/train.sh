#!/bin/bash
DATASET_ROOT="/home/xhb/datasets/cvpmvsnet_data/dtu-train-128/"

# Logging
CKPT_DIR="./checkpoints/"
LOG_DIR="./logs/"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train.py \
	--info "un-cvp-mvs-2021-04-20" \
	--mode "train" \
	--dataset_root $DATASET_ROOT \
	--imgsize  128 \
	--nsrc 6 \
	--nscale 2 \
	--epochs 40 \
	--lr 0.001 \
	--lrepochs "10,12,14,20:2" \
	--batch_size 32 \
	--loadckpt '' \
	--logckptdir $CKPT_DIR \
	--loggingdir $LOG_DIR \
	--resume 0  \
	--summarydir "summary" \
	--interval_scale 1.06 \
	--summary_freq 250 \
	--save_freq 2500 \
	--seg_clusters 4 \
	--w_seg 0.01 \
	--w_aug 0.01
