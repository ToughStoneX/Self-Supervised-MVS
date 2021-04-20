# -*- coding: utf-8 -*-
# @Time    : 2020/04/16 16:59
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : argParser
# @Software: PyCharm

import argparse


def getArgsParser():
    parser = argparse.ArgumentParser(description='Cost Volume Pyramid Based Depth Inference for Multi-View Stereo')

    # General settings
    parser.add_argument('--info', default='None', help='Info about current run')
    parser.add_argument('--mode', default='train', help='train or test ro validation', choices=['train', 'test', 'eval'])

    # Data settings
    parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
    parser.add_argument('--trainlist', default=None, help='train list')
    parser.add_argument('--dataset', default='dtu', help='select dataset')
    parser.add_argument('--dataset_root', help='path to dataset root')
    parser.add_argument('--imgsize', type=int, default=128, choices=[128, 1200], help='height of input image')
    parser.add_argument('--nsrc', type=int, default=2, help='number of src views to use')
    parser.add_argument('--nscale', type=int, default=5, help='number of scales to use')

    # Training settings
    parser.add_argument('--epochs', type=int, default=28, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lrepochs', type=str, default="10,12,14,20:2",
                        help='epoch ids to downscale lr and the downscale rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
    parser.add_argument('--summary_freq', type=int, default=200, help='print and summary frequency')
    parser.add_argument('--save_freq', type=int, default=2000, help='save checkpoint frequency')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--loss_function', default='sl1', help='which loss function to use', choices=['sl1', 'mse'])

    # Checkpoint settings
    parser.add_argument('--loadckpt', type=str, default='', help='load a specific checkpoint')
    parser.add_argument('--logckptdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
    parser.add_argument('--loggingdir', default='./logs/', help='the directory to save logging outputs')
    parser.add_argument('--resume', type=int, default=0, help='continue to train the model')

    # Evaluation settings
    parser.add_argument('--outdir', default='./outputs/debug/', help='the directory to save depth outputs')
    parser.add_argument('--eval_visualizeDepth', type=int, default=1)
    parser.add_argument('--eval_prob_filtering', type=int, default=0)
    parser.add_argument('--eval_prob_threshold', type=float, default=0.99)
    parser.add_argument('--eval_shuffle', type=int, default=0)

    parser.add_argument('--interval_scale', type=float, default=1.06)
    parser.add_argument('--summarydir', type=str, default='summary')

    parser.add_argument('--seg_clusters', type=int, default=4, help='cluster centers for unsupervised co-segmentation')
    parser.add_argument('--w_seg', type=float, default=0.01, help='weight for segments reprojection loss')
    parser.add_argument('--w_aug', type=float, default=0.01, help='weight of augment loss')

    return parser


def checkArgs(args):
    # Check if the settings is valid
    assert args.mode in ["train", "eval", "test"]
    if args.resume:
        assert len(args.loadckpt) == 0
    if args.loadckpt:
        assert args.resume is 0