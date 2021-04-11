#!/bin/bash

python eval_dense.py \
                --model "mvsnet" \
                --dataset "dtu_yao_eval" \
                --testpath "/home/xhb/datasets/mvsnet_data/test_data" \
                --testlist "lists/dtu/test.txt" \
                --batch_size 1 \
                --numdepth 256 \
                --interval_scale 1.06 \
                --loadckpt "./log-2020-07-01/model_00060000.ckpt" \
                --refine False \
                --outdir "./outputs_dense" \
                --gpu_device "1"
