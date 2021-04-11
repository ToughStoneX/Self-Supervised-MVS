# -*- coding: utf-8 -*-
# @Time    : 2020/05/26 22:17
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : dtu_yao2
# @Software: PyCharm

from torch.utils.data import Dataset
import numpy as np
import os
import math
from PIL import Image
import cv2
from torchvision import transforms
import torch

from datasets.data_io import *
from config import args


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


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

        self.transform_aug = transforms.Compose([
            transforms.ColorJitter(brightness=1, contrast=1, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            RandomGamma(min_gamma=0.5, max_gamma=2.0, clip_image=True)
        ])
        self.transform_seg = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def center_image(self, img):
        """ normalize image input """
        img = img.astype(np.float32)
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + 0.00000001)

    def scale_camera(self, cam, scale=1):
        """ resize input in order to produce sampled depth map """
        new_cam = np.copy(cam)
        # focal:
        new_cam[1][0][0] = cam[1][0][0] * scale
        new_cam[1][1][1] = cam[1][1][1] * scale
        # principle point:
        new_cam[1][0][2] = cam[1][0][2] * scale
        new_cam[1][1][2] = cam[1][1][2] * scale
        return new_cam

    def scale_mvs_camera(self, cams, scale=1):
        """ resize input in order to produce sampled depth map """
        for view in range(args.view_num):
            cams[view] = self.scale_camera(cams[view], scale=scale)
        return cams

    def scale_image(self, image, scale=1, interpolation='linear'):
        """ resize image using cv2 """
        if interpolation == 'linear':
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if interpolation == 'nearest':
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def scale_mvs_input(self, images, cams, depth_image=None, scale=1):
        """ resize input to fit into the memory """
        for view in range(args.view_num):
            images[view] = self.scale_image(images[view], scale=scale)
            cams[view] = self.scale_camera(cams[view], scale=scale)

        if depth_image is None:
            return images, cams
        else:
            depth_image = self.scale_image(depth_image, scale=scale, interpolation='nearest')
            return images, cams, depth_image

    def crop_mvs_input(self, images, cams, depth_image=None):
        """ resize images and cameras to fit the network (can be divided by base image size) """

        # crop images and cameras
        for view in range(args.view_num):
            h, w = images[view].shape[0:2]
            new_h = h
            new_w = w
            if new_h > args.max_h:
                new_h = args.max_h
            else:
                new_h = int(math.ceil(h / args.base_image_size) * args.base_image_size)
            if new_w > args.max_w:
                new_w = args.max_w
            else:
                new_w = int(math.ceil(w / args.base_image_size) * args.base_image_size)
            start_h = int(math.ceil((h - new_h) / 2))
            start_w = int(math.ceil((w - new_w) / 2))
            finish_h = start_h + new_h
            finish_w = start_w + new_w
            images[view] = images[view][start_h:finish_h, start_w:finish_w]
            cams[view][1][0][2] = cams[view][1][0][2] - start_w
            cams[view][1][1][2] = cams[view][1][1][2] - start_h

        # crop depth image
        if not depth_image is None:
            depth_image = depth_image[start_h:finish_h, start_w:finish_w]
            return images, cams, depth_image
        else:
            return images, cams

    def mask_depth_image(self, depth_image, min_depth, max_depth):
        """ mask out-of-range pixel to zero """
        ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
        ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
        depth_image = np.expand_dims(depth_image, 2)
        return depth_image

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

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

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

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        ###### read input data ######
        images = []
        images_aug = []
        images_seg = []
        cams = []
        proj_matrices = []
        intrinsics_list = []
        extrinsics_list = []
        # for view in range(self.nviews):
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)
            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

            image = self.center_image(cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB))
            image_aug = self.center_image(self.read_img_aug(img_filename))
            image_seg = self.read_img_seg(img_filename)
            cam = self.load_cam(open(proj_mat_filename))
            cam[1][3][1] = cam[1][3][1] * args.interval_scale
            images.append(image)
            images_seg.append(image_seg)
            images_aug.append(image_aug)
            cams.append(cam)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                         dtype=np.float32)
                depth_image = self.read_depth(depth_filename)
                # mask out-of-range depth pixels (in a relaxed range)
                depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                depth_end = cams[0][1, 3, 0] + (args.numdepth - 2) * (cams[0][1, 3, 1])
                depth_image =self.mask_depth_image(depth_image, depth_start, depth_end)
                mask = self.read_img(mask_filename)
        images = np.stack(images).transpose([0, 3, 1, 2])
        images_aug = np.stack(images_aug).transpose([0, 3, 1, 2])
        images_seg = np.stack(images_seg)
        cams = np.stack(cams)
        proj_matrices = np.stack(proj_matrices)
        intrinsics_list = np.stack(intrinsics_list)
        extrinsics_list = np.stack(extrinsics_list)
        return {"imgs": images,
                "imgs_aug": images_aug,
                "imgs_seg": images_seg,
                "proj_matrices": proj_matrices,
                "intrinsics": intrinsics_list,
                "extrinsics": extrinsics_list,
                "mask": mask,
                "cams": cams,
                "depth": depth_image,
                "depth_values": depth_values,
                "depth_start": depth_start}


if __name__ == "__main__":
    datapath = "D:\\BaiduNetdiskDownload\\mvsnet\\training_data\\dtu_training"
    listfile = "E:\\PycharmProjects\\un_mvsnet_pytorch\\lists\\dtu\\train.txt"
    train_dataset = MVSDataset(datapath, listfile, "train", nviews=3, ndepths=192, interval_scale=1.06)
    print('dataset length: {}'.format(len(train_dataset)))
    item = train_dataset[50]
    print(item.keys())
    print("imgs", item["imgs"].shape)
    print("depth", item["depth"].shape)
    print("cams", item["cams"].shape)


    # # some testing code, just IGNORE it
    # dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
    #                      3, 128)
    # item = dataset[50]
    #
    # dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/val.txt', 'val', 3,
    #                      128)
    # item = dataset[50]
    #
    # dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/test.txt', 'test', 5,
    #                      128)
    # item = dataset[50]
    #
    # # test homography here
    # print(item.keys())
    # print("imgs", item["imgs"].shape)
    # print("depth", item["depth"].shape)
    # print("depth_values", item["depth_values"].shape)
    # print("mask", item["mask"].shape)
    #
    # ref_img = item["imgs"][0].transpose([1, 2, 0])[::4, ::4]
    # src_imgs = [item["imgs"][i].transpose([1, 2, 0])[::4, ::4] for i in range(1, 5)]
    # ref_proj_mat = item["proj_matrices"][0]
    # src_proj_mats = [item["proj_matrices"][i] for i in range(1, 5)]
    # mask = item["mask"]
    # depth = item["depth"]
    #
    # height = ref_img.shape[0]
    # width = ref_img.shape[1]
    # xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    # print("yy", yy.max(), yy.min())
    # yy = yy.reshape([-1])
    # xx = xx.reshape([-1])
    # X = np.vstack((xx, yy, np.ones_like(xx)))
    # D = depth.reshape([-1])
    # print("X", "D", X.shape, D.shape)
    #
    # X = np.vstack((X * D, np.ones_like(xx)))
    # X = np.matmul(np.linalg.inv(ref_proj_mat), X)
    # X = np.matmul(src_proj_mats[0], X)
    # X /= X[2]
    # X = X[:2]
    #
    # yy = X[0].reshape([height, width]).astype(np.float32)
    # xx = X[1].reshape([height, width]).astype(np.float32)
    # import cv2
    #
    # warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
    # warped[mask[:, :] < 0.5] = 0
    #
    # cv2.imwrite('../tmp0.png', ref_img[:, :, ::-1] * 255)
    # cv2.imwrite('../tmp1.png', warped[:, :, ::-1] * 255)
    # cv2.imwrite('../tmp2.png', src_imgs[0][:, :, ::-1] * 255)