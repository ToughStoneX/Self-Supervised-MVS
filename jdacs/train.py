# -*- coding: utf-8 -*-
# @Time    : 2020/05/21 22:37
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : train
# @Software: PyCharm


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
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
import gc
import sys
import datetime

from models.mvsnet import MVSNet, mvsnet_loss
from losses.unsup_seg_loss import *
from losses.unsup_loss import *
from models.augmentations import random_image_mask, aug_loss
from utils import *
from config import args, device

cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create logger for mode "train" and "testall"
if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)
    # wandb.init(name='mvsnet pytorch', notes='baseline', config=args, sync_tensorboard=True)

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.view_num, args.numdepth, args.interval_scale)
test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.view_num, args.numdepth, args.interval_scale)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = MVSNet(refine=args.refine).to(device)
# if args.mode in ["train", "test"]:
#     model = nn.DataParallel(model)
model = nn.DataParallel(model)
# model_loss = mvsnet_loss
criterion = UnSupLoss().to(device)
criterion_seg = UnSupSegLoss(args).to(device)
criterion_seg = nn.DataParallel(criterion_seg)
criterion_aug = aug_loss
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

seg_viz = SegVisualizer(K=args.seg_clusters)

# load parameters
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def adjust_parameters(epoch):
    if epoch == 2-1:
        args.w_aug = 2 * args.w_aug
    elif epoch == 4-1:
        args.w_aug = 2 * args.w_aug
    elif epoch == 6-1:
        args.w_aug = 2 * args.w_aug
    elif epoch == 8-1:
        args.w_aug = 2 * args.w_aug


# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)
    global_step = 0

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        # global_step = len(TrainImgLoader) * epoch_idx
        adjust_parameters(epoch_idx)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step += 1
            # global_step = epoch_idx * len(TrainImgLoader) + batch_idx
            do_summary = global_step % args.summary_freq == 0 or global_step == 1
            loss, scalar_outputs, image_outputs, segment_loss, depth_est = train_sample(sample, do_summary)
            if do_summary or global_step == 1:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs
            del image_outputs
            augment_loss, scalar_outputs, image_outputs = train_sample_aug(sample, depth_est, do_summary)
            if do_summary or global_step == 1:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs
            del image_outputs
            print(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, standard loss = {:.3f}, '
                'segment loss = {:.3f}, augment_loss: {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                                                              loss, criterion.unsup_loss, segment_loss, augment_loss,
                                                              time.time() - start_time))

            if global_step % args.val_freq == 0:
                # testing
                avg_test_scalars = DictAverageMeter()
                for batch_idx, sample in enumerate(TestImgLoader):
                    start_time = time.time()
                    # global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                    # do_summary = global_step % args.summary_freq == 0
                    loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
                    # if do_summary:
                    #     save_scalars(logger, 'test', scalar_outputs, global_step)
                    #     save_images(logger, 'test', image_outputs, global_step)
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs
                    del image_outputs
                    print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                             batch_idx,
                                                                                             len(TestImgLoader), loss,
                                                                                             time.time() - start_time))
                save_scalars(logger, 'test', avg_test_scalars.mean(), global_step)
                print("avg_test_scalars:", avg_test_scalars.mean())

                torch.save({
                    'epoch': epoch_idx,
                    'iter': global_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:08}.ckpt".format(args.logdir, global_step))


def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs
        del image_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                    time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    print("final", avg_test_scalars)


def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    depth_gt = depth_gt.squeeze(-1)
    mask = sample_cuda["mask"]
    depth_interval = sample_cuda["cams"][:, 0, 1, 3, 1]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    # imgs, cams, depth, depth_start
    standard_loss = criterion(sample_cuda["imgs"], sample_cuda["cams"], depth_est)
    segment_loss, ref_seg, view_segs = criterion_seg(sample_cuda["imgs_seg"], sample_cuda["cams"], depth_est)
    segment_loss = torch.mean(segment_loss)
    loss = standard_loss + segment_loss * args.w_seg
    loss.backward()
    optimizer.step()

    scalar_outputs = {"loss": loss, "standard_loss": standard_loss,
                      "reconstr_loss": criterion.reconstr_loss, "ssim_loss": criterion.ssim_loss,
                      "smooth_loss": criterion.smooth_loss,
                      "segment_loss": segment_loss}

    # ref_seg: [B, H, W, C]
    ref_seg_idx = torch.argmax(ref_seg, dim=3)  # [B, H, W]
    ref_seg_viz = seg_viz.convert_heatmap(ref_seg_idx.cpu().numpy())  # [B, H, W, C]

    image_outputs = {"depth_est": depth_est * mask, "depth_gt": sample["depth"].squeeze(dim=-1),
                     "ref_img": sample["imgs"][:, 0], "ref_seg": ref_seg_viz,
                     "mask": sample["mask"]}

    for i in range(view_segs.size(1)):
        view_seg = view_segs[:, i]
        view_seg_idx = torch.argmax(view_seg, dim=3)  # [B, H, W]
        view_seg_viz = seg_viz.convert_heatmap(view_seg_idx.cpu().numpy())  # [B, H, W, C]
        image_outputs["src_img_{}".format(i+1)] = sample["imgs"][:, 1+i]
        image_outputs["src_seg_{}".format(i+1)] = view_seg_viz

    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
        scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
        scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
        scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)
        scalar_outputs["mae"] = non_zero_mean_absolute_diff(depth_gt, depth_est, depth_interval)
        scalar_outputs["less_one_accuracy"] = less_one_percentage(depth_gt, depth_est, depth_interval)
        scalar_outputs["less_three_accuracy"] = less_three_percentage(depth_gt, depth_est, depth_interval)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs, tensor2float(segment_loss), depth_est.clone().detach()



def train_sample_aug(sample, depth_est, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    # depth_gt = depth_gt.squeeze(-1)
    mask = sample_cuda["mask"]
    # depth_interval = sample_cuda["cams"][:, 0, 1, 3, 1]

    # batch_size = sample_cuda["imgs"].size(0)
    imgs_aug = sample_cuda["imgs_aug"]
    ref_img = imgs_aug[:, 0]

    # step 1 正常训练
    # outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    # depth_est = outputs["depth"]
    # # 重建损失
    # standard_loss = criterion(sample_cuda["imgs"], sample_cuda["cams"], depth_est)
    # loss = standard_loss
    # print('standard_loss: {}'.format(standard_loss))
    # standard_loss.backward(retain_graph=True)

    # step 2 数据增强一致性训练
    # 数据增强(光照调整/gamma校正，随机噪声，随机mask)
    ref_img, filter_mask = random_image_mask(ref_img, filter_size=(ref_img.size(2)//3, ref_img.size(3)//3))
    imgs_aug[:, 0] = ref_img
    outputs_aug = model(imgs_aug, sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est_aug = outputs_aug["depth"]
    # 数据增强损失
    filter_mask = F.interpolate(filter_mask, scale_factor=0.25)
    filter_mask = filter_mask[:, 0, :, :]
    # print('depth_est_aug: {} depth_est: {}'.format(depth_est_aug.shape, depth_est.shape))
    augment_loss = criterion_aug(depth_est_aug, depth_est, filter_mask)
    # print('augment_loss: {}'.format(augment_loss))
    augment_loss = augment_loss * args.w_aug
    augment_loss.backward()

    optimizer.step()

    scalar_outputs = {"augment_loss": augment_loss}
    # print('depth_est: {}'.format(depth_est.shape))
    # print('depth_gt: {}'.format(sample["depth"].shape))
    # print('ref_img: {}'.format(sample["imgs"][:, 0].shape))
    # print('mask: {}'.format(sample["mask"].shape))
    image_outputs = {"depth_est_aug": depth_est_aug * mask * filter_mask}

    return tensor2float(augment_loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    depth_gt = depth_gt.squeeze(-1)
    mask = sample_cuda["mask"]
    depth_interval = sample_cuda["cams"][:, 0, 1, 3, 1]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    # loss = model_loss(depth_est, depth_gt, mask)
    # imgs, cams, depth, depth_start
    loss = criterion(sample_cuda["imgs"], sample_cuda["cams"], depth_est)

    scalar_outputs = {"loss": loss, "reconstr_loss": criterion.reconstr_loss, "ssim_loss": criterion.ssim_loss,
                      "smooth_loss": criterion.smooth_loss}
    image_outputs = {"depth_est": depth_est * mask, "depth_gt": sample["depth"].squeeze(dim=-1),
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]}
    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask

    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)
    scalar_outputs["mae"] = non_zero_mean_absolute_diff(depth_gt, depth_est, depth_interval)
    scalar_outputs["less_one_accuracy"] = less_one_percentage(depth_gt, depth_est, depth_interval)
    scalar_outputs["less_three_accuracy"] = less_three_percentage(depth_gt, depth_est, depth_interval)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "profile":
        profile()