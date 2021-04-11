# -*- coding: utf-8 -*-
# @Time    : 2020/06/25 10:18
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : unsup_seg_loss
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import args, device
from models.seg_dff import SegDFF
from losses.modules import *
from losses.homography import *


def compute_seg_loss(warped_seg, ref_seg, mask):
    mask = mask.repeat(1, 1, 1, warped_seg.size(3))
    # print('mask: {}'.format(mask.shape))
    # print('warped_seg: {}'.format(warped_seg.shape))
    # print('ref_seg: {}'.format(ref_seg.shape))
    warped_seg_filtered = warped_seg[mask > 0.5]
    ref_seg_filtered = ref_seg[mask > 0.5]
    # print('warped_seg_filtered: {}'.format(warped_seg_filtered.shape))
    # print('ref_seg_filtered: {}'.format(ref_seg_filtered.shape))
    warped_seg_filtered_flatten = warped_seg_filtered.contiguous().view(-1, warped_seg.size(3))  # [B * H * W, C]
    ref_seg_filtered_flatten = ref_seg_filtered.contiguous().view(-1, ref_seg.size(3))  # [B * H * W, C]
    ref_seg_filtered_flatten = torch.argmax(ref_seg_filtered_flatten, dim=1) # [B, H, W]
    loss = F.cross_entropy(warped_seg_filtered_flatten, ref_seg_filtered_flatten, size_average=True)
    return loss


class UnSupSegLoss(nn.Module):
    def __init__(self, args):
        super(UnSupSegLoss, self).__init__()
        self.seg_model = SegDFF(K=args.seg_clusters, max_iter=50)

    def forward(self, imgs, cams, depth):
        # print('imgs: {}'.format(imgs.shape))
        seg_maps = self.seg_model(imgs)  # # [batch_size, num_views, K, height, width]
        seg_maps = seg_maps.permute(0, 1, 4, 2, 3)  # [1, 7, 14, 14, 4]
        # print('seg_maps: {}'.format(seg_maps.shape))

        seg_maps = torch.unbind(seg_maps, 1)
        cams = torch.unbind(cams, 1)
        height, width = depth.size(1), depth.size(2)
        num_views = len(seg_maps)

        ref_seg = seg_maps[0]
        ref_seg = F.interpolate(ref_seg, size=(height, width), mode='bilinear')
        ref_seg = ref_seg.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]

        warped_seg_list = []
        mask_list = []
        reprojection_losses = []
        view_segs = []
        for view in range(1, num_views):
            view_seg = seg_maps[view]
            view_seg = F.interpolate(view_seg, size=(height, width), mode='bilinear')
            view_seg = view_seg.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            view_cam = cams[view]
            view_segs.append(view_seg)

            warped_seg, mask = inverse_warping(view_seg, ref_cam, view_cam, depth)
            warped_seg_list.append(warped_seg)
            mask_list.append(mask)
            # warped_seg: [B, H, W, C]
            # mask: [B, H, W]
            reprojection_losses.append(compute_seg_loss(warped_seg, ref_seg, mask))
        reproj_seg_loss = sum(reprojection_losses) * 1.0

        # self.ref_seg = ref_seg
        view_segs = torch.stack(view_segs, dim=1)

        return reproj_seg_loss, ref_seg, view_segs




