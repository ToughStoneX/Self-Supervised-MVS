# -*- coding: utf-8 -*-
# @Time    : 2020/05/26 22:26
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : config
# @Software: PyCharm

import os
import torch
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='A PyTorch Implementation of MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--dataset', default='dtu_yao2', help='select dataset')
parser.add_argument('--trainpath', default=None, help='train datapath')
parser.add_argument('--testpath', default=None, help='test datapath')
parser.add_argument('--trainlist', default=None, help='train list')
parser.add_argument('--testlist', default=None, help='test list')
parser.add_argument('--epochs', type=int, default=8, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="4,6,7:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.6, help='the number of depth values')
# parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./log_tb', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--summary_freq', type=int, default=500, help='print and summary frequency')
parser.add_argument('--val_freq', type=int, default=5000, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=6, metavar='S', help='random seed')
parser.add_argument('--gpu_device', type=str, default='0,1,2,3', help='GPU')
parser.add_argument('--smooth_lambda', type=float, default=1.0, help='weight for smooth term')
parser.add_argument('--view_num', type=int, default=7, help='Number of images (1 ref image and view_num - 1 view images).')
parser.add_argument('--refine', type=str2bool, default=False, help='Whether using a ResNet to refine the depth map')
parser.add_argument('--seg_clusters', type=int, default=4, help='cluster centers for unsupervised co-segmentation')
parser.add_argument('--w_seg', type=float, default=0.01, help='weight for segments reprojection loss')
parser.add_argument('--w_aug', type=float, default=0.01, help='weight of augment loss')

parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--display', action='store_true', help='display depth images and masks')
parser.add_argument('--volume_loss', default=False, help='Use cost volume to generate ref image from src images')

# parse arguments and check
args = parser.parse_args()
if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath

# device
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')