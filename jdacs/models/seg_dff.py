# -*- coding: utf-8 -*-
# @Time    : 2020/06/24 16:46
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : seg_dff
# @Software: PyCharm

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import random


EPSILON = 1e-7

# object function for nmf
def approximation_error(V, W, H, square_root=True):
    # Frobenius Norm
    return torch.norm(V - torch.mm(W, H))


def multiplicative_update_step(V, W, H, update_h=None, VH=None, HH=None):
    # update operation for W
    if VH is None:
        assert HH is None
        Ht = torch.t(H)  # [k, m] --> [m, k]
        VH = torch.mm(V, Ht)  # [n, m] x [m, k] --> [n, k]
        HH = torch.mm(H, Ht)  # [k, m] x [m, k] --> [k, k]

    WHH = torch.mm(W, HH) # [n, k] x [k, k] --> [n, k]
    WHH[WHH == 0] = EPSILON
    W *= VH / WHH

    if update_h:
        # update operation for H (after updating W)
        Wt = torch.t(W)  # [n, k] --> [k, n]
        WV = torch.mm(Wt, V)  # [k, n] x [n, m] --> [k, m]
        WWH = torch.mm(torch.mm(Wt, W), H)  #
        WWH[WWH == 0] = EPSILON
        H *= WV / WWH
        VH, HH = None, None

    return W, H, VH, HH


def NMF(V, k, W=None, H=None, random_seed=None, max_iter=200, tol=1e-4, cuda=True, verbose=False):
    if verbose:
        start_time = time.time()

    # scale = math.sqrt(V.mean() / k)
    scale = torch.sqrt(V.mean() / k)

    if random_seed is not None:
        if cuda:
            current_random_seed = torch.cuda.initial_seed()
            torch.cuda.manual_seed(random_seed)
        else:
            current_random_seed = torch.initial_seed()
            torch.manual_seed(random_seed)

    if W is None:
        if cuda:
            W = torch.cuda.FloatTensor(V.size(0), k).normal_()
        else:
            W = torch.randn(V.size(0), k)
        W *= scale  # [n, k]

    update_H = True
    if H is None:
        if cuda:
            H = torch.cuda.FloatTensor(k, V.size(1)).normal_()
        else:
            H = torch.randn(k, V.size(1))
        H *= scale  # [k, m]
    else:
        update_H = False

    if random_seed is not None:
        if cuda:
            torch.cuda.manual_seed(current_random_seed)
        else:
            torch.manual_seed(current_random_seed)

    W = torch.abs(W)
    H = torch.abs(H)

    error_at_init = approximation_error(V, W, H, square_root=True)
    previous_error = error_at_init

    VH = None
    HH = None
    for n_iter in range(max_iter):
        W, H, VH, HH = multiplicative_update_step(V, W, H, update_h=update_H, VH=VH, HH=HH)
        if tol > 0 and n_iter % 10 == 0:
            error = approximation_error(V, W, H, square_root=True)
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    if verbose:
        print('Exited after {} iterations. Total time: {} seconds'.format(n_iter+1, time.time()-start_time))
    return W, H


class SegDFF(nn.Module):
    def __init__(self, K, max_iter=50):
        super(SegDFF, self).__init__()
        self.K = K
        self.max_iter = max_iter
        self.net = models.vgg19(pretrained=True)
        del self.net.features._modules['36'] # delete redundant layers to save memory

    def forward(self, imgs):
        # imgs: [batch_size, num_views, 3, height, width]
        batch_size = imgs.size(0)
        heatmaps = []
        for b in range(batch_size):
            imgs_b = imgs[b]
            with torch.no_grad():
                # h, w = imgs_b.size(2), imgs_b.size(3)
                imgs_b = F.interpolate(imgs_b, size=(224, 224), mode='bilinear', align_corners=False)
                features = self.net.features(imgs_b)
                flat_features = features.permute(0, 2, 3, 1).contiguous().view(-1, features.size(1))
                W, _ = NMF(flat_features, self.K, random_seed=1, cuda=True, max_iter=self.max_iter, verbose=False)
                # print(torch.isnan(W))
                isnan = torch.sum(torch.isnan(W).float())
                while isnan > 0:
                    # 注：NMF有时求解会失败，W矩阵全部会nan值，在反向传播时是无效的。一旦出现求解失败的情况，就重新初始化随机参数进行求解。
                    print('nan detected. trying to resolve the nmf.')
                    W, _ = NMF(flat_features, self.K, random_seed=random.randint(0, 255), cuda=True, max_iter=self.max_iter, verbose=False)
                    isnan = torch.sum(torch.isnan(W).float())
                heatmap = W.view(features.size(0), features.size(2), features.size(3), self.K)
                # heatmap = F.softmax(heatmap, dim=-1)
                # heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
                # heatmap = torch.argmax(heatmap, dim=3)  # [num_views, height, width]
                heatmaps.append(heatmap)
        heatmaps = torch.stack(heatmaps, dim=0)  # [batch_size, num_views, K, height, width]
        heatmaps.requires_grad = False
        return heatmaps



