#!/bin/bash

# dataset
DATASET_ROOT="/home/xhb/datasets/cvpmvsnet_data/dtu-test-1200/"
# checkpoint
LOAD_CKPT_DIR="./checkpoints/model_00030000.ckpt"
# logging
LOG_DIR="./logs/"
# output
OUT_DIR="./outputs/"

CUDA_VISIBLE_DEVICES="0" python test.py \
	--info "un-cvp-mvs-2021-04-20" \
	--mode "test" \
	--dataset_root $DATASET_ROOT \
	--imgsize 1200 \
	--nsrc 4 \
	--nscale 5 \
	--batch_size 1 \
	--loadckpt $LOAD_CKPT_DIR \
	--loggingdir $LOG_DIR \
	--outdir $OUT_DIR
