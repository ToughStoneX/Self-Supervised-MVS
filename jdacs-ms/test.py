# -*- coding: utf-8 -*-
# @Time    : 2020/04/17 21:48
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : eval
# @Software: PyCharm

import os
import time
import logging
import sys
import numpy as np
import errno
from PIL import Image
import gc
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset.dtu import DTUDataset
from models.network import CVPMVSNet
from argParser import getArgsParser
from utils import tocuda


cudnn.benchmark = True

# Arg parser
parser = getArgsParser()
args = parser.parse_args()
assert args.mode == "test"

# logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
log_path = args.loggingdir+args.info.replace(" ","_")+"/"
if not os.path.isdir(args.loggingdir):
    os.mkdir(args.loggingdir)
if not os.path.isdir(log_path):
    os.mkdir(log_path)
log_name = log_path + curTime + '.log'
logfile = log_name
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fileHandler = logging.FileHandler(logfile, mode='a')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
logger.info("Logger initialized.")
logger.info("Writing logs to file:"+logfile)

settings_str = "All settings:\n"
line_width = 30
for k,v in vars(args).items():
    settings_str += '{0}: {1}\n'.format(k,v)
logger.info(settings_str)

# Run CVP-MVSNet to save depth maps and confidence maps
def save_depth():
    # dataset
    test_dataset = DTUDataset(args, logger)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    # model
    model = CVPMVSNet(args)
    model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint
    logger.info("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'], strict=False)

    # infer
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            start_time = time.time()
            sample_cuda = tocuda(sample)
            torch.cuda.empty_cache()
            outputs = model(
                sample_cuda["ref_img"].float(),
                sample_cuda["src_imgs"].float(),
                sample_cuda["ref_intrinsics"].float(),
                sample_cuda["src_intrinsics"].float(),
                sample_cuda["ref_extrinsics"].float(),
                sample_cuda["src_extrinsics"].float(),
                sample_cuda["depth_min"].float(),
                sample_cuda["depth_max"].float()
            )

            # parse output
            depth_est_list = outputs["depth_est_list"]
            depth_est = depth_est_list[0].data.cpu().numpy()
            prob_confidence = outputs["prob_confidence"].data.cpu().numpy()

            del sample_cuda
            filenames = sample["filename"]
            logger.info('Iter {}/{}, time = {:.3f}'.format(batch_idx, len(test_loader), time.time() - start_time))

            # save depth maps and confidence maps
            for filename, est_depth, photometric_confidence in zip(filenames, depth_est, prob_confidence):
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, est_depth)
                write_depth_img(depth_filename + ".png", est_depth)
                # Save prob maps
                save_pfm(confidence_filename, photometric_confidence)

            gc.collect()


def save_pfm(filename, image, scale=1):

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


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


if __name__ == '__main__':
    save_depth()


