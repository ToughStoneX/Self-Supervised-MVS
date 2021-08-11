# JDACS-MS

## About

This is the training code of the **JDACS-MS** framework proposed in our AAAI-21 paper: "*Self-supervised Multi-view Stereo via Effective Co-Segmentation and Data-Augmentation*" [[paper]](https://www.aaai.org/AAAI21Papers/AAAI-2549.XuH.pdf).

JDACS-MS supports the training of multi-stage MVSNet which possesses a coarse-to-fine principle. It is noted that [CVP-MVSNet](https://github.com/JiayuYANG/CVP-MVSNet) is utilized as the backbone in default. You can also replace the backbone network with your own customized model.

## How to use?

### Environment
 - Python 3.6 (Anaconda) 
 - Pytorch 1.1.0
 - 4 to 8 GPUs with 11G memories, such as 1080 Ti or 2080 Ti.

### Training
 - Download the preprocessed DTU training dataset of [CVP-MVSNet](https://github.com/JiayuYANG/CVP-MVSNet) \[ [Google Cloud](https://drive.google.com/file/d/1_Nuud3lRGaN_DOkeTNOvzwxYa2z2YRbX/view) \]. (Note: This processed partition is different from the original one provided by MVSNet, whose resolution of images is downsampled to 128 x 160.)
 - Edit the training settings in `train.sh`.
   - `DATASET_ROOT`: the path of your training dataset. 
   - `CKPT_DIR`: the path to save the checkpoint weights of your model during training.
   - `LOG_DIR`: the path to save log files.
   - `CUDA_VISIBLE_DEVICES=xxx`: the ids of adopted GPUs. You should modify it according to the available GPUs in your server.
   - `--epochs`: the total epochs for training.
   - `--batch_size`: batch size during training.
   - `--logdir`: the name of the checkpoint dir.
   - `--w_aug`: weight for data-augmentation consistency loss. The default value is 0.01.
   - `--w_seg`: weight for co-segmentation consistency loss. The default value is 0.01.
   - `--seg_clusters`: the cluster centroids for NMF (Co-segmentation). The default value is 4.
 - Train the model by running `./train.sh`.

### Evaluating
 - Download the preprocessed DTU testing data of [CVP-MVSNet](https://github.com/JiayuYANG/CVP-MVSNet) \[ [Google Cloud](https://drive.google.com/file/d/1rX0EXlUL4prRxrRu2DgLJv2j7-tpUD4D/view) \].
 - Edit the testing setting in 'test.sh`:
   - `DATASET_ROOT`: the path of the testing dataset.
   - `LOAD_CKPT_DIR`: the path of the checkpoint file.
   - `LOG_DIR`: the directory to save the log file.
   - `OUT_DIR`: the output directory.
   - `CUDA_VISIBLE_DEVICES=x`: the id of selected GPU to conduct the depth inference process.
 - Generate the depth maps by running `./test.sh`.

### Fusion
 - To fuse the generated per-view depth maps to a 3D point cloud, we utilize the code of [fusibile](https://github.com/kysucix/fusibile) for depth fusion.
 - To build the binary executable file from [fusibile](https://github.com/kysucix/fusibile), please follow the following steps:
   - Enter the `fusion/fusibile` directory of this project.
   - Check the gpu architecture in your server and modify the corresponding settings in `CMakeList.txt`:
     - If 1080 Ti GPU with a computation capability of 6.0 is used, please add: `set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=sm_60)`.
     - If 2080 Ti GPU with a computation capability of 7.5 is used, please add: `set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=sm_75)`.
     - For other GPUs, please check the computation capability of your GPU in: https://developer.nvidia.com/zh-cn/cuda-gpus.
   - Create the directory by running `mkdir build`.
   - Enter the created directory, `cd build`.
   - Configure the CMake setting, `cmake ..`.
   - Build the project, `make`.
   - Then you can find the binary executable file named as `fusibile` in the `build` directory.
  - Go back to the `fusion/` directory and edit the fusion setting in `fusion.sh`:
    - `DTU_TEST_ROOT`: the path of the testing dataset.
    - `FUSIBILE_EXE_PATH`: the path of the binary executable `fusibile` file.
    - `--prob_threshold`/`--disp_threshold`/`--num_consistent`: hyperparamters for depth fusion.
 - Run the fusion code, `./fusion.sh`.
 - Move the fused 3D point cloud to the same folder, `python arange.py`. You can find the reconstructed 3D ply files in the directory of `outputs/mvsnet_0.4_0.25`.

### Benchmark
 - To reproduce the quantitative performance of the model, we can use the official code provided by [DTU](http://roboimagedata.compute.dtu.dk/?page_id=36). The original codes are implemented in Matlab and requires huge time for calculating the evaluation metrics. To accelerate this process, we also provide a parallel version of the evaluation code in `matlab_eval/dtu` of this repo.
 - Download the [Sample Set.zip](roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) and [Points.zip](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip) from [DTU](http://roboimagedata.compute.dtu.dk/?page_id=36)'s official website. Decompress the zip files and arange the ground truth point clouds following the official instructions of [DTU](http://roboimagedata.compute.dtu.dk/?page_id=36).
 - Edit the path settings in `ComputeStat_web.m` and `BaseEvalMain_web.m`.
   - The `datapath` should be changed according to the path of your data. For example, `dataPath='/home/xhb/dtu_eval/SampleSet/MVS Data/';`
 - Enter the `matlab_eval/dtu` directory and run the matlab evaluation code, `./run.sh`. The results will be presented in a few hours. The time consumption is up to the available threads enabled in the Matlab environment. 

## Note

- The training process of CVP-MVSNet in JDACS-MS is time-consuming. It is suggested that 8 GPUs are used to conduct the training process, whereas 4 GPUs may take several days for training.
- The default hyperparameters `--prob_threshold`/`--disp_threshold`/`--num_consistent` may not be the best configuration, which requires further adjustment. It may be better to manually adjust these hyperparameters.
- The adopted model checkpoint with iteration of 30000 steps in `./test.sh` can also be alternated with other checkpoints, which may produce better performance.
