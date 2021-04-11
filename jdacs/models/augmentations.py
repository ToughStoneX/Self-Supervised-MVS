# -*- coding: utf-8 -*-
# @Time    : 2020/06/15 14:38
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : augmentations
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import ImageFilter

from config import device


class Augmentor(nn.Module):
    def __init__(self):
        super(Augmentor, self).__init__()
        self.transform = get_transform()

    def forward(self, imgs):
        # imgs = imgs.permute(1, 0, 2, 3, 4)
        # print(imgs.shape)
        imgs = [torch.stack(self.transform(im_b), dim=0) for im_b in imgs]
        imgs = torch.stack(imgs, dim=0)
        # imgs = imgs.permute(1, 0, 2, 3, 4)
        ref_img = imgs[:, 0]
        # print(imgs.shape)
        # print(ref_img.shape)
        h, w = ref_img.size(2), ref_img.size(3)
        ref_img, filter_mask = random_image_mask(ref_img, filter_size=(h // 4, w // 4))
        # print(imgs.shape)
        # print(ref_img.shape)
        imgs[:, 0] = ref_img
        return imgs, filter_mask


def get_transform():
    transform = []
    transform.append(ToPILImage())
    transform.append(ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0))
    transform.append(ToTensor())
    transform.append(RandomGamma(min_gamma=0.7, max_gamma=2.0, clip_image=True))
    return transforms.Compose(transform)


class ToTensor(transforms.ToTensor):
    def __call__(self, imgs):
        return [super(ToTensor, self).__call__(im) for im in imgs]


class ToPILImage(transforms.ToPILImage):
    def __call__(self, imgs):
        return [super(ToPILImage, self).__call__(im) for im in imgs]


class ColorJitter(transforms.ColorJitter):
    def __call__(self, imgs):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return [transform(im) for im in imgs]


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

    def __call__(self, imgs):
        # gamma = self.get_params(self._min_gamma, self._max_gamma)
        # return [self.adjust_gamma(im, gamma, self._clip_image) for im in imgs]
        res = []
        for im in imgs:
            gamma = self.get_params(self._min_gamma, self._max_gamma)
            res.append(self.adjust_gamma(im, gamma, self._clip_image))
        return res


class RandomGaussianBlur():
    def __init__(self, p, max_k_sz):
        self.p = p
        self.max_k_sz = max_k_sz

    def __call__(self, imgs):
        if np.random.random() < self.p:
            radius = np.random.uniform(0, self.max_k_sz)
            imgs = [im.filter(ImageFilter.GaussianBlur(radius)) for im in imgs]
        return imgs


def random_image_mask(img, filter_size):
    '''

    :param img: [B x 3 x H x W]
    :param crop_size:
    :return:
    '''
    fh, fw = filter_size
    _, _, h, w = img.size()

    if fh == h and fw == w:
        return img, None

    x = np.random.randint(0, w - fw)
    y = np.random.randint(0, h - fh)
    filter_mask = torch.ones_like(img)    # B x 3 x H x W
    filter_mask[:, :, y:y+fh, x:x+fw] = 0.0    # B x 3 x H x W
    img = img * filter_mask    # B x 3 x H x W
    return img, filter_mask


def aug_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)


if __name__ == '__main__':
    aug = Augmentor()

    from datasets.dtu_yao2 import MVSDataset
    datapath = "D:\\BaiduNetdiskDownload\\mvsnet\\training_data\\dtu_training"
    listfile = "E:\\PycharmProjects\\un_mvsnet_pytorch\\lists\\dtu\\train.txt"
    train_dataset = MVSDataset(datapath, listfile, "train", nviews=3, ndepths=192, interval_scale=1.06)
    print('dataset length: {}'.format(len(train_dataset)))
    item = train_dataset[50]
    print(item.keys())
    print("imgs", item["imgs"].shape)
    print("depth", item["depth"].shape)
    print("cams", item["cams"].shape)
    print("mask", item["mask"].shape)

    imgs_tensor = torch.from_numpy(item["imgs"])
    depth_tensor = torch.from_numpy(item["depth"])
    cams_tensor = torch.from_numpy(item["cams"])
    mask_tensor = torch.from_numpy(item["mask"])

    imgs_tensor = imgs_tensor.unsqueeze(dim=0)
    ref_img_tensor = imgs_tensor[0].unsqueeze(dim=0)
    depth_tensor = depth_tensor.unsqueeze(dim=0).squeeze(dim=3)
    ref_cam_tensor = cams_tensor[0].unsqueeze(dim=0)
    mask_tensor = mask_tensor.unsqueeze(dim=0)
    # print(depth_tensor.shape)

    # ref_img_tensor = F.interpolate(ref_img_tensor, scale_factor=0.25)

    h, w = depth_tensor.size(1), depth_tensor.size(2)

    imgs_filtered, filter_mask = aug(imgs_tensor.clone())
    # ref_img_filtered, depth_filtered, mask_filtered, filter_mask = random_apply_mask(ref_img_tensor, depth_tensor, mask_tensor,
    #                                                                                  filter_size=(h//2, w//2))


    # print(ref_img_tensor.shape)
    ref_img_np = ref_img_tensor[0, 0].permute(1, 2, 0).cpu().numpy()
    depth_np = depth_tensor[0].cpu().numpy()
    mask_np = mask_tensor[0].cpu().numpy()
    # ref_img_filtered_np = ref_img_filtered[0].permute(1, 2, 0).cpu().numpy()
    # depth_filtered_np = depth_filtered[0].cpu().numpy()
    # mask_filtered_np = mask_filtered[0].cpu().numpy()

    from matplotlib import pyplot as plt
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(ref_img_np)
    plt.subplot(2,3,2)
    plt.imshow(depth_np)
    plt.subplot(2,3,3)
    plt.imshow(mask_np)
    plt.subplot(2, 3, 4)
    # plt.imshow(ref_img_filtered_np)
    plt.imshow(imgs_filtered[0, 0].permute(1, 2, 0).cpu().numpy())
    plt.subplot(2, 3, 5)
    # plt.imshow(depth_filtered_np)
    plt.imshow(imgs_filtered[0, 1].permute(1, 2, 0).cpu().numpy())
    plt.subplot(2, 3, 6)
    # plt.imshow(mask_filtered_np)
    plt.imshow(imgs_filtered[0, 2].permute(1, 2, 0).cpu().numpy())
    plt.show()

