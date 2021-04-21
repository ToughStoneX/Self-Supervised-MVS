# -*- coding: utf-8 -*-
# @Time    : 2020/04/16 13:56
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : dtu
# @Software: PyCharm

import numpy as np
import os
import sys
from PIL import Image
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import torch

from dataset.utils import *
from dataset.data_path import *


class RandomGamma():
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        adjusted = torch.pow(image, gamma)
        if clip_image:
            adjusted.clamp_(0.0, 1.0)
        return adjusted

    def __call__(self, img):
        gamma = self.get_params(self._min_gamma, self._max_gamma)
        return self.adjust_gamma(img, gamma, self._clip_image)


class DTUDataset(Dataset):
    def __init__(self, args, logger=None):
        # Initializing the dataloader
        super(DTUDataset, self).__init__()

        # Parse input
        self.args = args
        self.data_root = self.args.dataset_root
        if self.args.mode == 'train' or self.args.mode == 'test':
            self.scan_list_file = getScanListFile(self.data_root, self.args.mode)
            self.pair_list_file = getPairListFile(self.data_root, self.args.mode)
        else:  # eval
            self.scan_list_file = getScanListFile(self.data_root, 'test')
            self.pair_list_file = getPairListFile(self.data_root, 'test')
        self.logger = logger
        if logger == None:
            import logging
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
            consoleHandler = logging.StreamHandler(sys.stdout)
            consoleHandler.setFormatter(formatter)
            self.logger.addHandler(consoleHandler)
            self.logger.info("File logger not configured, only writing logs to stdout.")
        self.logger.info("Initiating dataloader for our pre-processed DTU dataset.")
        self.logger.info("Using dataset:" + self.data_root + self.args.mode + "/")

        if self.args.mode == 'train' or self.args.mode == 'test':
            self.metas = self.build_list(self.args.mode)
        else: # eval
            self.metas = self.build_list('test')
        self.logger.info("Dataloader initialized.")

        self.transform_seg = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_aug = transforms.Compose([
            transforms.ColorJitter(brightness=1, contrast=1, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            RandomGamma(min_gamma=0.5, max_gamma=2.0, clip_image=True)
        ])

    def build_list(self, mode):
        # Build the item meta list
        metas = []

        # Read scan list
        scan_list = readScanList(self.scan_list_file, self.args.mode, self.logger)

        # Read pairs list
        for scan in scan_list:
            with open(self.pair_list_file) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    if mode == 'train':
                        for light_idx in range(7):
                            metas.append((scan, ref_view, src_views, light_idx))
                    elif mode == 'test':
                        metas.append((scan, ref_view, src_views, 3))
        self.logger.info("Done. metas:" + str(len(metas)))
        return metas

    def center_image(self, img):
        """ normalize image input """
        img = img.astype(np.float32)
        if img.shape[0] == 1200:
            img = img[:1184,:1600,:]
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + 0.00000001)

    def load_cam(self, file, interval_scale=1):
        """ read camera txt file """
        cam = np.zeros((2, 4, 4))
        words = file.read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                cam[0][i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                cam[1][i][j] = words[intrinsic_index]

        if len(words) == 29:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = 256
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 30:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 31:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = 0
            cam[1][3][1] = 0
            cam[1][3][2] = 0
            cam[1][3][3] = 0

        return cam

    def read_img_seg(self, filename):
        img = Image.open(filename)
        return self.transform_seg(img)

    def read_img_aug(self, filename):
        img = Image.open(filename)
        img = self.transform_aug(img)
        # print(img.shape)
        img = img.permute(1, 2, 0)
        img = img * 255
        return img.numpy()

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views, light_idx = meta

        assert self.args.nsrc <= len(src_views)

        self.logger.debug("Getting Item:\nscan:" + str(scan) + "\nref_view:" + str(ref_view) + "\nsrc_view:" + str(
            src_views) + "\nlight_idx" + str(light_idx))

        ref_img = []
        src_imgs = []
        src_imgs_aug = []
        ref_depths = []
        ref_depth_mask = []
        ref_intrinsics = []
        src_intrinsics = []
        ref_extrinsics = []
        src_extrinsics = []
        depth_min = []
        depth_max = []
        imgs = []
        cams = []
        images_seg = []

        ## 1. Read images
        # ref image
        ref_img_file = getImageFile(self.data_root, self.args.mode, scan, ref_view, light_idx)
        # ref_img = read_img(ref_img_file)
        ref_img = self.center_image(cv2.cvtColor(cv2.imread(ref_img_file), cv2.COLOR_BGR2RGB))
        imgs.append(ref_img)
        ref_img_seg = self.read_img_seg(ref_img_file)
        images_seg.append(ref_img_seg)
        ref_img_aug = self.center_image(self.read_img_aug(ref_img_file))

        # src image(s)
        for i in range(self.args.nsrc):
            src_img_file = getImageFile(self.data_root, self.args.mode, scan, src_views[i], light_idx)
            # src_img = read_img(src_img_file)
            src_img = self.center_image(cv2.cvtColor(cv2.imread(src_img_file), cv2.COLOR_BGR2RGB))
            src_imgs.append(src_img)
            imgs.append(src_img)
            src_img_seg = self.read_img_seg(src_img_file)
            images_seg.append(src_img_seg)
            src_img_aug = self.center_image(self.read_img_aug(src_img_file))
            src_imgs_aug.append(src_img_aug)

        ## 2. Read camera parameters
        cam_file = getCameraFile(self.data_root, self.args.mode, ref_view)
        ref_intrinsics, ref_extrinsics, depth_min, depth_max = read_cam_file(cam_file)
        cam = self.load_cam(open(cam_file))
        cam[1][3][1] = cam[1][3][1] * self.args.interval_scale
        cams.append(cam)
        for i in range(self.args.nsrc):
            cam_file = getCameraFile(self.data_root, self.args.mode, src_views[i])
            intrinsics, extrinsics, depth_min_tmp, depth_max_tmp = read_cam_file(cam_file)
            src_intrinsics.append(intrinsics)
            src_extrinsics.append(extrinsics)
            cam = self.load_cam(open(cam_file))
            cam[1][3][1] = cam[1][3][1] * self.args.interval_scale
            cams.append(cam)

        ## 3. Read Depth Maps
        if self.args.mode == "train":
            imgSize = self.args.imgsize
            nscale = self.args.nscale

            # Read depth map of same size as input image first.
            depth_file = getDepthFile(self.data_root, self.args.mode, scan, ref_view)
            ref_depth = read_depth(depth_file)
            depth_frame_size = (ref_depth.shape[0], ref_depth.shape[1])
            frame = np.zeros(depth_frame_size)
            frame[:ref_depth.shape[0], :ref_depth.shape[1]] = ref_depth
            ref_depths.append(frame)

            # Downsample the depth for each scale.
            ref_depth = Image.fromarray(ref_depth)
            original_size = np.array(ref_depth.size).astype(int)

            for scale in range(1, nscale):
                new_size = (original_size / (2**scale)).astype(int)
                down_depth = ref_depth.resize((new_size), Image.BICUBIC)
                frame = np.zeros(depth_frame_size)
                down_np_depth = np.array(down_depth)
                frame[:down_np_depth.shape[0], :down_np_depth.shape[1]] = down_np_depth
                ref_depths.append(frame)
        elif self.args.mode == "eval":
            # Read depth map of same size as input image first.
            depth_file = getDepthFile(self.data_root, "test", scan, ref_view)
            ref_depth = read_depth(depth_file)
            # depth_frame_size = (ref_depth.shape[0], ref_depth.shape[1])
            # frame = np.zeros(depth_frame_size)
            # frame[:ref_depth.shape[0], :ref_depth.shape[1]] = ref_depth
            ref_depths.append(ref_depth)

        # Orgnize output and return
        sample = {}
        sample["ref_img"] = np.moveaxis(np.array(ref_img), 2, 0)
        sample["ref_img_aug"] = np.moveaxis(np.array(ref_img_aug), 2, 0)
        sample["src_imgs"] = np.moveaxis(np.array(src_imgs), 3, 1)
        sample["src_imgs_aug"] = np.moveaxis(np.array(src_imgs_aug), 3, 1)
        sample["ref_intrinsics"] = np.array(ref_intrinsics)
        sample["src_intrinsics"] = np.array(src_intrinsics)
        sample["ref_extrinsics"] = np.array(ref_extrinsics)
        sample["src_extrinsics"] = np.array(src_extrinsics)
        sample["depth_min"] = depth_min
        sample["depth_max"] = depth_max
        sample["imgs"] = np.stack(imgs).transpose([0, 3, 1, 2])
        sample["cams"] = np.stack(cams)
        sample["imgs_seg"] = np.stack(images_seg)
        # sample["coseg"] = np.load('./coseg_maps/{}.npz'.format(idx))['arr_0'][0]
        # sample["coseg"] = np.load('./coseg_ft/{}.npz'.format(idx))['arr_0']

        # print('sample["cams"]: {}'.format(sample["cams"]))

        if self.args.mode == 'train':
            sample["ref_depths"] = np.array(ref_depths, dtype=float)
            sample["ref_depth_mask"] = np.array(ref_depth_mask)
        elif self.args.mode == 'test' or self.args.mode == 'eval':
            sample["ref_depths"] = np.array(ref_depths, dtype=float)
            sample["ref_depth_mask"] = np.array(ref_depth_mask)
            sample["filename"] = scan + '/{}/' + '{:0>8}'.format(ref_view) + "{}"

        return sample


