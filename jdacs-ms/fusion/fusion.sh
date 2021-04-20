#!/bin/bash

DTU_TEST_ROOT="/home/xhb/datasets/mvsnet_data/test_data"
DEPTH_FOLDER="../outputs/"
FUSIBILE_EXE_PATH="./fusibile/build/fusibile"

CUDA_VISIBLE_DEVICES="0" python depthfusion.py \
        --dtu_test_root $DTU_TEST_ROOT \
        --depth_folder $DEPTH_FOLDER \
        --out_folder "fused_0.4_0.25" \
        --fusibile_exe_path $FUSIBILE_EXE_PATH \
        --prob_threshold 0.4 \
        --disp_threshold 0.25 \
        --num_consistent 3
