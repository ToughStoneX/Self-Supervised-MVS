# -*- coding: utf-8 -*-
# @Time    : 2020/05/29 20:19
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : unsup_loss
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import args, device
from losses.modules import *
from losses.homography import *


class UnSupLoss(nn.Module):
    def __init__(self):
        super(UnSupLoss, self).__init__()
        self.ssim = SSIM()

    def forward(self, imgs, cams, depth):
        # print('imgs: {}'.format(imgs.shape))
        # print('cams: {}'.format(cams.shape))
        # print('depth: {}'.format(depth.shape))

        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_views = len(imgs)

        ref_img = imgs[0]
        ref_img = F.interpolate(ref_img, scale_factor=0.25, mode='bilinear')
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]
        # print('ref_cam: {}'.format(ref_cam.shape))

        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0

        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]
            # print('view_cam: {}'.format(view_cam.shape))
            view_img = F.interpolate(view_img, scale_factor=0.25, mode='bilinear')
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            # warp view_img to the ref_img using the dmap of the ref_img
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)
            warped_img_list.append(warped_img)
            mask_list.append(mask)

            reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

            # SSIM loss##
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))

        ##smooth loss##
        self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, args.smooth_lambda)

        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses).permute(1,2,3,4,0)
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
        top_vals = torch.neg(top_vals)
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)

        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + 0.18 * self.smooth_loss
        # 按照un_mvsnet和M3VSNet的设置
        # self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15
        return self.unsup_loss


def non_zero_mean_absolute_diff(y_true, y_pred, interval):
    """ non zero mean absolute loss for one batch """
    batch_size = y_pred.shape[0]
    interval = interval.reshape(batch_size)
    mask_true = torch.ne(y_true, 0.0).float()
    denom = torch.sum(mask_true, dim=[1,2]) + 1e-7
    masked_abs_error = torch.abs(mask_true * (y_true - y_pred))
    masked_mae = torch.sum(masked_abs_error, dim=[1,2])
    masked_mae = torch.sum((masked_mae.float() / interval.float()) / denom.float())
    return masked_mae


def less_one_percentage(y_true, y_pred, interval):
    """ less one accuracy for one batch """
    batch_size = y_pred.shape[0]
    height = y_pred.shape[1]
    width = y_pred.shape[2]
    interval = interval.reshape(batch_size)
    mask_true = torch.ne(y_true, 0.0).float()
    denom = torch.sum(mask_true) + 1e-7
    interval_image = interval.reshape(batch_size, 1, 1).repeat(1, height, width)
    # print('y_true: {}'.format(y_true.shape))
    # print('y_pred: {}'.format(y_pred.shape))
    abs_diff_image = torch.abs(y_true.float() - y_pred.float()) / interval_image.float()
    less_three_image = mask_true * torch.le(abs_diff_image, 1.0).float()
    return torch.sum(less_three_image) / denom


def less_three_percentage(y_true, y_pred, interval):
    """ less three accuracy for one batch """
    batch_size = y_pred.shape[0]
    height = y_pred.shape[1]
    width = y_pred.shape[2]
    interval = interval.reshape(batch_size)
    mask_true = torch.ne(y_true, 0.0).float()
    denom = torch.sum(mask_true) + 1e-7
    interval_image = interval.reshape(batch_size, 1, 1).repeat(1, height, width)
    abs_diff_image = torch.abs(y_true.float() - y_pred.float()) / interval_image.float()
    less_three_image = mask_true * torch.le(abs_diff_image, 3.0).float()
    return torch.sum(less_three_image) / denom






