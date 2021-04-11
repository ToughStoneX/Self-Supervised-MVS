# -*- coding: utf-8 -*-
# @Time    : 2020/05/26 20:34
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : eval
# @Software: PyCharm

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from models.mvsnet import MVSNet
from utils import *
import sys
from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
import errno

from config import args, device

cudnn.benchmark = True

# parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse. May be different from the original implementation')
# parser.add_argument('--model', default='mvsnet', help='select model')
#
# parser.add_argument('--dataset', default='dtu_yao_eval2', help='select dataset')
# parser.add_argument('--testpath', help='testing data path')
# parser.add_argument('--testlist', help='testing scan list')
#
# parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
# parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
# parser.add_argument('--interval_scale', type=float, default=0.8, help='the depth interval scale')
#
# parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
# parser.add_argument('--outdir', default='./outputs', help='output dir')
# parser.add_argument('--display', action='store_true', help='display depth images and masks')
# parser.add_argument('--gpu_device', type=str, default='0', help='GPU')

# parse arguments and check
# args = parser.parse_args()
# print("argv:", sys.argv[1:])
# print_args(args)

# device
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    intrinsics[:2, :] /= 4
    intrinsics[0] *= 1152 / 1600
    intrinsics[1] *= 864 / 1200
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    # img = Image.open(filename)
    img = Image.open(filename).resize((1152, 864), Image.BILINEAR)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data


def write_depth_img(filename,depth):

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    image = Image.fromarray((depth-500)/2).convert("L")
    image.save(filename)
    return 1


# run MVS model to save depth maps and confidence maps
def save_depth():
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model
    # model = MVSNet(refine=False, mode='test')
    model = MVSNet(refine=args.refine).to(device)
    model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            # resize output
            depth = outputs["depth"]
            photometric_confidence = outputs["photometric_confidence"]
            depth = F.interpolate(depth.unsqueeze(1), size=(1200, 1600)).squeeze(1)
            photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1), size=(1200, 1600)).squeeze(1)
            outputs["depth"] = depth
            outputs["photometric_confidence"] = photometric_confidence

            outputs = tensor2numpy(outputs)
            del sample_cuda
            print('Iter {}/{}'.format(batch_idx, len(TestImgLoader)))
            filenames = sample["filename"]

            # save depth maps and confidence maps
            for filename, depth_est, photometric_confidence in zip(filenames, outputs["depth"],
                                                                   outputs["photometric_confidence"]):
                # print('depth_est: {}'.format(depth_est.shape))
                # 深度图上采样
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, depth_est)
                # save confidence maps
                save_pfm(confidence_filename, photometric_confidence)
                write_depth_img(depth_filename + ".png", depth_est)


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    save_depth()

