# -*- coding: utf-8 -*-
# @Time    : 2020/04/16 14:31
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : modules
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=True),
        nn.LeakyReLU(0.1)
    )


def conditionIntrinsics(intrinsics, img_shape, fp_shapes):
    # Pre-condition intrinsics according to feature pyramid shape.

    # Calculate downsample ratio for each level of feture pyramid
    down_ratios = []
    for fp_shape in fp_shapes:
        down_ratios.append(img_shape[2] / fp_shape[2])

    # condition intrinsics
    intrinsics_out = []
    for down_ratio in down_ratios:
        intrinsics_tmp = intrinsics.clone()
        intrinsics_tmp[:, :2, :] = intrinsics_tmp[:, :2, :] / down_ratio
        intrinsics_out.append(intrinsics_tmp)

    return torch.stack(intrinsics_out).permute(1, 0, 2, 3)  # [B, nScale, 3, 3]


def calInitDepthInterval(ref_in, src_in, ref_ex, src_ex, pixel_interval):
    return 165  # The mean depth interval calculated on 4-1 interval setting...


def calSweepingDepthHypo(ref_in, src_in, ref_ex, src_ex, depth_min, depth_max, nhypothesis_init=48):
    # Batch
    batchSize = ref_in.shape[0]
    # print('depth_max: {}'.format(depth_max))
    # print('depth_min: {}'.format(depth_min))
    depth_range = depth_max[0] - depth_min[0]
    depth_interval_mean = depth_range / (nhypothesis_init - 1)
    # Make sure the number of depth hypothesis has a factor of 2
    assert nhypothesis_init % 2 == 0
    depth_hypos = torch.range(depth_min[0], depth_max[0], depth_interval_mean).unsqueeze(0)
    # Assume depth range is consistent in one batch.
    for b in range(1, batchSize):
        # depth_range = depth_max[b] - depth_min[b]
        depth_hypos = torch.cat(
            (depth_hypos, torch.range(depth_min[0], depth_max[0], depth_interval_mean).unsqueeze(0)), 0)
    return depth_hypos.cuda()


def homo_warping(src_feature, ref_in, src_in, ref_ex, src_ex, depth_hypos):
    # Apply homography warpping on one src feature map from src to ref view.

    batch, channels = src_feature.shape[0], src_feature.shape[1]
    num_depth = depth_hypos.shape[1]
    height, width = src_feature.shape[2], src_feature.shape[3]

    with torch.no_grad():
        # 内参矩阵乘上外参矩阵计算单应性矩阵
        src_proj = torch.matmul(src_in, src_ex[:, 0:3, :])
        ref_proj = torch.matmul(ref_in, ref_ex[:, 0:3, :])
        last = torch.tensor([[[0,0,0,1.0]]]).repeat(len(src_in),1,1).cuda()
        src_proj = torch.cat((src_proj, last), 1)
        ref_proj = torch.cat((ref_proj, last), 1)

        # 根据两个相机的单应性矩阵计算ref视角图像与src视角图像之前的旋转矩阵和平移矩阵
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        # 根据图像的像素位置计算旋转后的关系
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_feature.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_feature.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_hypos.view(batch, 1, num_depth,
                                                                                           1)  # [B, 3, Ndepth, H*W]
        # 加上平移向量
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # 除以z坐标，进行归一化
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        # 换算采样索引
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    warped_src_fea = F.grid_sample(src_feature, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    return warped_src_fea


def calDepthHypo(netArgs, ref_depths, ref_intrinsics, src_intrinsics, ref_extrinsics, src_extrinsics,
                 depth_min, depth_max, level):
    ## Calculate depth hypothesis maps for refine steps
    nhypothesis_init = 48
    d = 4
    pixel_interval = 1

    nBatch = ref_depths.shape[0]
    height = ref_depths.shape[1]
    width = ref_depths.shape[2]
    # print('ref_depths: {}'.format(ref_depths.shape))

    with torch.no_grad():
        ref_depths = ref_depths
        ref_intrinsics = ref_intrinsics.double()
        src_intrinsics = src_intrinsics.squeeze(1).double()
        ref_extrinsics = ref_extrinsics.double()
        src_extrinsics = src_extrinsics.squeeze(1).double()

        interval_maps = []
        depth_hypos = ref_depths.unsqueeze(1).repeat(1, d * 2, 1, 1).double()
        for batch in range(nBatch):
            # 计算相机坐标系下的xyz坐标值
            xx, yy = torch.meshgrid([torch.arange(0, width).cuda(), torch.arange(0, height).cuda()])

            xxx = xx.reshape([-1]).double()
            yyy = yy.reshape([-1]).double()

            X = torch.stack([xxx, yyy, torch.ones_like(xxx)], dim=0)

            D1 = torch.transpose(ref_depths[batch, :, :], 0, 1).reshape(
                [-1]).double()  # Transpose before reshape to produce identical results to numpy and matlab version.
            D2 = D1 + 1

            # 乘上深度（z坐标，对齐尺度）
            X1 = X * D1
            X2 = X * D2
            # 乘上ref相机内参，得到相机坐标系下的xyz坐标
            ray1 = torch.matmul(torch.inverse(ref_intrinsics[batch]), X1)
            ray2 = torch.matmul(torch.inverse(ref_intrinsics[batch]), X2)

            # 乘上ref相机外参，校正到世界坐标系
            X1 = torch.cat([ray1, torch.ones_like(xxx).unsqueeze(0).double()], dim=0)
            X1 = torch.matmul(torch.inverse(ref_extrinsics[batch]), X1)
            X2 = torch.cat([ray2, torch.ones_like(xxx).unsqueeze(0).double()], dim=0)
            X2 = torch.matmul(torch.inverse(ref_extrinsics[batch]), X2)

            # 乘上src相机外参，转到src相机坐标系
            X1 = torch.matmul(src_extrinsics[batch][0], X1)
            X2 = torch.matmul(src_extrinsics[batch][0], X2)

            # 乘上src相机内参，转到src图像坐标系
            X1 = X1[:3]
            X1 = torch.matmul(src_intrinsics[batch][0], X1)
            # 除以z坐标（深度）转换到图像坐标系
            X1_d = X1[2].clone()
            X1 /= X1_d

            X2 = X2[:3]
            X2 = torch.matmul(src_intrinsics[batch][0], X2)
            X2_d = X2[2].clone()
            X2 /= X2_d

            k = (X2[1] - X1[1]) / (X2[0] - X1[0])
            b = X1[1] - k * X1[0]

            theta = torch.atan(k)
            X3 = X1 + torch.stack(
                [torch.cos(theta) * pixel_interval, torch.sin(theta) * pixel_interval, torch.zeros_like(X1[2, :])],
                dim=0)

            A = torch.matmul(ref_intrinsics[batch], ref_extrinsics[batch][:3, :3])
            tmp = torch.matmul(src_intrinsics[batch][0], src_extrinsics[batch][0, :3, :3])
            A = torch.matmul(A, torch.inverse(tmp))

            tmp1 = X1_d * torch.matmul(A, X1)
            tmp2 = torch.matmul(A, X3)

            M1 = torch.cat([X.t().unsqueeze(2), tmp2.t().unsqueeze(2)], dim=2)[:, 1:, :]
            M2 = tmp1.t()[:, 1:]
            # print('M1: {} M2: {}'.format(M1.shape, M2.shape))
            # print('torch.inverse(M1): {}'.format(torch.inverse(M1).shape))
            # print('M2.unsqueeze(2): {}'.format(M2.unsqueeze(2).shape))
            tmp1 = torch.inverse(M1)
            tmp2 = M2.unsqueeze(2)
            ans = torch.matmul(tmp1, tmp2)
            # ans = torch.bmm(tmp1, tmp2)
            delta_d = ans[:, 0, 0]

            interval_maps = torch.abs(delta_d).mean().repeat(ref_depths.shape[2], ref_depths.shape[1]).t()

            for depth_level in range(-d, d):
                depth_hypos[batch, depth_level + d, :, :] += depth_level * interval_maps

            # print("Calculated:")
            # print(interval_maps[0,0])

            # pdb.set_trace()

        return depth_hypos.float()  # Return the depth hypothesis map from statistical interval setting.


def proj_cost(settings, ref_feature, src_feature, level, ref_in, src_in, ref_ex, src_ex, depth_hypos):
    ## Calculate the cost volume for refined depth hypothesis selection
    batch, channels = ref_feature.shape[0], ref_feature.shape[1]
    num_depth = depth_hypos.shape[1]
    height, width = ref_feature.shape[2], ref_feature.shape[3]
    nSrc = len(src_feature)

    volume_sum = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
    volume_sq_sum = volume_sum.pow_(2)

    for src in range(settings.nsrc):
        with torch.no_grad():
            src_proj = torch.matmul(src_in[:, src, :, :], src_ex[:, src, 0:3, :])
            ref_proj = torch.matmul(ref_in, ref_ex[:, 0:3, :])
            last = torch.tensor([[[0, 0, 0, 1.0]]]).repeat(len(src_in), 1, 1).cuda()
            src_proj = torch.cat((src_proj, last), 1)
            ref_proj = torch.cat((ref_proj, last), 1)

            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj[:, :3, :3]
            trans = proj[:, :3, 3:4]

            y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=ref_feature.device),
                                   torch.arange(0, width, dtype=torch.float32, device=ref_feature.device)])
            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(height * width), x.view(height * width)
            xyz = torch.stack((x, y, torch.ones_like(x)))
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)
            rot_xyz = torch.matmul(rot, xyz)

            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_hypos.view(batch, 1, num_depth,
                                                                                               height * width)  # [B, 3, Ndepth, H*W]
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)
            proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
            proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
            proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
            proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)
            grid = proj_xy
        warped_src_fea = F.grid_sample(src_feature[src][level], grid.view(batch, num_depth * height, width, 2),
                                       mode='bilinear',
                                       padding_mode='zeros')
        warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

        volume_sum = volume_sum + warped_src_fea
        volume_sq_sum = volume_sq_sum + warped_src_fea.pow_(2)
    cost_volume = volume_sq_sum.div_(settings.nsrc + 1).sub_(volume_sum.div_(settings.nsrc + 1).pow_(2))

    if settings.mode == "test":
        del volume_sum
        del volume_sq_sum
        torch.cuda.empty_cache()

    return cost_volume


# MVSNet modules
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth


def depth_regression_refine(prob_volume, depth_hypothesis):
    return torch.sum(prob_volume * depth_hypothesis, 1)