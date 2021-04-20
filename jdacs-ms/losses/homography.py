# -*- coding: utf-8 -*-
# @Time    : 2020/05/28 22:30
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : homography
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


# def get_homography(left_cam, right_cam, depth_num, depth_start, depth_interval):
#     '''
#     :param left_cam: [batch_size, 2, 4, 4]
#     :param right_cam: [batch_size, 2, 4, 4]
#     :param depth_num:
#     :param depth_start:
#     :param depth_interval:
#     :return:
#     '''
#     # cameras (K, R, t)
#     R_left = left_cam[:, 0, :3, :3].unsqueeze(1)  # [batch_size, 1, 3, 3]
#     R_right = right_cam[:, 0, :3, :3].unsqueeze(1)  # [batch_size, 1, 3, 3]
#     t_left = left_cam[:, 0, :3, 3:].unsqueeze(1)  # [batch_size, 1, 3, 1]
#     t_right = right_cam[:, 0, :3, 3:].unsqueeze(1)  # [batch_size, 1, 3, 1]
#     K_left = left_cam[:, 1, :3, :3].unsqueeze(1)  # [batch_size, 1, 3, 3]
#     K_right = right_cam[:, 1, :3, :3].unsqueeze(1)  # [batch_size, 1, 3, 3]
#
#     print('R_left: {}'.format(R_left.shape))
#     print('R_right: {}'.format(R_right.shape))
#     print('t_left: {}'.format(t_left.shape))
#     print('t_right: {}'.format(t_right.shape))
#     print('K_left: {}'.format(K_left.shape))
#     print('K_right: {}'.format(K_right.shape))
#
#     # depth
#     depth = depth_start + torch.arange(depth_num) * depth_interval
#     # preparation
#     num_depth = depth.shape[0]
#     K_left_inv = torch.inverse(torch.squeeze(K_left, dim=1))  # [batch_size, 3, 1]
#     R_left_trans = torch.squeeze(R_left, dim=1).permute(0, 2, 1)  # [batch_size, 3, 3]
#     R_right_trans = torch.squeeze(R_right, dim=1).permute(0, 2, 1)  # [batch_size, 3, 3]
#
#     fronto_direction = torch.squeeze(R_left, dim=1)[:, 2:, :]  # [batch_size, 1, 3]
#
#     c_left = torch.matmul(R_left_trans, torch.squeeze(t_left, dim=1))  # [batch_size, 3, 1]
#     c_right = torch.matmul(R_right_trans, torch.squeeze(t_right, dim=1))  # [batch_size, 3, 1]
#     c_relative = c_right - c_left  # [batch_size, 3, 1]
#
#     # compute
#     batch_size = R_left.shape[0]
#     temp_vec = torch.matmul(c_relative, fronto_direction)  # [batch_size, 3, 3]
#     depth_mat = depth.reshape(batch_size, num_depth, 1, 1).repeat(1, 1, 3, 3)  # [batch_size, num_depth, 3, 3]
#     temp_vec = temp_vec.unsqueeze(dim=1).repeat(1, num_depth, 1, 1)  # [batch_size, num_depth, 3, 3]
#
#     middle_mat0 = torch.eye(3, device=device).unsqueeze(dim=0).unsqueeze(dim=1).repeat(batch_size, num_depth, 1, 1) - \
#         temp_vec / depth_mat  # [batch_size, num_depth, 3, 3]
#     middle_mat1 = torch.matmul(R_left_trans, K_left_inv).unsqueeze(dim=1).repeat(1, num_depth, 1, 1)  # [batch_size, num_depth, 3, 1]
#     middle_mat2 = torch.matmul(middle_mat0, middle_mat1)    # [batch_size, num_depth, 3, 1]
#
#     homographies = torch.matmul(K_right.repeat(1, num_depth, 1, 1),
#                                 torch.matmul(R_right.repeat(1, num_depth, 1, 1), middle_mat2))
#     return homographies
#
#
# def get_pixel_grids(height, width):
#     x_linspace = torch.linspace(0.5, width-0.5, width).to(device)
#     y_linspace = torch.linspace(0.5, height-0.5, height).to(device)
#     x_coordinates, y_coordinates = torch.meshgrid([x_linspace, y_linspace])
#     x_coordinates = x_coordinates.reshape(-1).unsqueeze(dim=0)
#     y_coordinates = y_coordinates.reshape(-1).unsqueeze(dim=0)
#     ones = torch.ones_like(x_coordinates).to(device)
#     indices_grid = torch.cat([x_coordinates, y_coordinates, ones], dim=0)
#     return indices_grid
#
#
# def repeat_int(x, num_repeats):
#     ones = torch.ones([1, num_repeats], device=device).int()
#     x = x.reshape(-1, 1)
#     x = torch.matmul(x, ones)
#     return x.reshape(-1)
#
#
# def th_flatten(x):
#     """Flatten tensor"""
#     return x.contiguous().view(-1)
#
#
# def th_gather_nd(x, coords):
#     '''
#     N-dimensional version of torch.gather
#     https://github.com/ncullen93/torchsample
#     '''
#     x = x.contiguous()
#     inds = coords.mv(torch.LongTensor(x.stride()))
#     x_gather = torch.index_select(th_flatten(x), 0, inds)
#     return x_gather
#
#
# def interpolate(image, x, y):
#     # [B, H, W, C]
#
#     image_shape = image.shape
#     batch_size = image_shape[0]
#     height = image_shape[1]
#     width = image_shape[2]
#
#     # image coordinate to pixel coordinate
#     x = x - 0.5
#     y = y - 0.5
#     x0 = torch.floor(x).int()
#     x1 = x0 + 1
#     y0 = torch.floor(y).int()
#     y1 = y0 + 1
#     max_y = int(height - 1)
#     max_x = int(width - 1)
#     x0 = torch.clamp(x0, 0, max_x)
#     x1 = torch.clamp(x1, 0, max_x)
#     y0 = torch.clamp(y0, 0, max_y)
#     y1 = torch.clamp(y1, 0, max_y)
#     b = repeat_int(torch.arange(batch_size).to(device), height*width)
#
#     indices_a = torch.stack([b, y0, x0], dim=1)
#     indices_b = torch.stack([b, y0, x1], dim=1)
#     indices_c = torch.stack([b, y1, x0], dim=1)
#     indices_d = torch.stack([b, y1, x1], dim=1)
#
#     pixel_values_a = th_gather_nd(image, indices_a)
#     pixel_values_b = th_gather_nd(image, indices_b)
#     pixel_values_c = th_gather_nd(image, indices_c)
#     pixel_values_d = th_gather_nd(image, indices_d)
#
#     x0, x1, y0, y1 = x0.float(), x1.float(), y0.float(), y1.float()
#     area_a = ((y1 - y) * (x1 - x))
#     area_b = ((y1 - y) * (x - x0))
#     area_c = ((y - y0) * (x1 - x))
#     area_d = ((y - y0) * (x - x0))
#     area_a, area_b, area_c, area_d = area_a.unsqueeze(1), area_b.unsqueeze(1), area_c.unsqueeze(1), area_d.unsqueeze(1)
#     output = area_a * pixel_values_a + area_b * pixel_values_b + area_c * pixel_values_c + area_d * pixel_values_d
#     return output
#
#
# def homography_warping(input_image, homography):
#     # input image [B, C, H, W]
#     input_image = input_image.permute(0, 2, 3, 1)
#     # [B, H, W, C]
#     image_shape = input_image.shape
#     batch_size = image_shape[0]
#     height = image_shape[2]
#     width = image_shape[3]
#
#     # turn homography to affine_mat of size (B, 2, 3) and div_mat of size (B, 1, 3)
#     affine_mat = homography[:, :2, :3]  # [B, 2, 3]
#     div_mat = homography[:, 2:3, :3]  # [B, 1, 3]
#
#     # generate pixel grids of size (B, 3, (W+1) x (H+1))
#     pixel_grids = get_pixel_grids(height, width)
#     # print(pixel_grids.shape)
#     pixel_grids = pixel_grids.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size, 3, -1)  # (B, 3, (W+1) x (H+1))
#
#     # affine + divide tranform, output (B, 2, (W+1) x (H+1))
#     print('affine_mat: {}'.format(affine_mat.shape))
#     print('pixel_grids: {}'.format(pixel_grids.shape))
#     grids_affine = torch.matmul(affine_mat.float(), pixel_grids.float())  # (B, 2, (W+1) x (H+1))
#     grids_div = torch.matmul(div_mat.float(), pixel_grids.float())  # (B, 1, (W+1) x (H+1))
#     grids_zero_add = torch.eq(grids_div, 0.0) * 1e-7  # handle div 0
#     grids_div = grids_div + grids_zero_add  # (B, 1, (W+1) x (H+1))
#     print('grids_div: {}'.format(grids_div.shape))
#     grids_div = grids_div.repeat(1, 2, 1)  # (B, 2, (W+1) x (H+1))
#     grids_div_warped = torch.div(grids_affine, grids_div)  # (B, 2, (W+1) x (H+1))
#     x_warped, y_warped = grids_div_warped[:, 0, :], grids_div_warped[:, 1, :]  # (B, (W+1) x (H+1))
#     x_warped_flatten, y_warped_flatten = x_warped.reshape(-1), y_warped.reshape(-1)
#
#     # interpolation
#     warped_image = interpolate(input_image, x_warped_flatten, y_warped_flatten)
#     warped_image = warped_image.reshape(batch_size, height, width, -1)
#
#     return warped_image.permute(0, 3, 1, 2)  # [B, C, H, W]


def inverse_warping(img, left_cam, right_cam, depth):
    # img: [batch_size, height, width, channels]

    # cameras (K, R, t)
    # print('left_cam: {}'.format(left_cam.shape))
    R_left = left_cam[:, 0:1, 0:3, 0:3]  # [B, 1, 3, 3]
    R_right = right_cam[:, 0:1, 0:3, 0:3]  # [B, 1, 3, 3]
    t_left = left_cam[:, 0:1, 0:3, 3:4]  # [B, 1, 3, 1]
    t_right = right_cam[:, 0:1, 0:3, 3:4]  # [B, 1, 3, 1]
    K_left = left_cam[:, 1:2, 0:3, 0:3]  # [B, 1, 3, 3]
    K_right = right_cam[:, 1:2, 0:3, 0:3]  # [B, 1, 3, 3]

    K_left = K_left.squeeze(1)  # [B, 3, 3]
    # print('left_cam: {}'.format(left_cam))
    # print('K_left: {}'.format(K_left))

    try:
        K_left_inv = torch.inverse(K_left)  # [B, 3, 3]
    except Exception as e:
        # print('K_left: {}'.format(K_left.cpu().numpy()))
        print(e)
        exit(-1)

    R_left_trans = R_left.squeeze(1).permute(0, 2, 1)  # [B, 3, 3]
    R_right_trans = R_right.squeeze(1).permute(0, 2, 1)  # [B, 3, 3]

    R_left = R_left.squeeze(1)
    t_left = t_left.squeeze(1)
    R_right = R_right.squeeze(1)
    t_right = t_right.squeeze(1)

    ## estimate egomotion by inverse composing R1,R2 and t1,t2
    R_rel = torch.matmul(R_right, R_left_trans)  # [B, 3, 3]
    t_rel = t_right - torch.matmul(R_rel, t_left)  # [B, 3, 1]
    ## now convert R and t to transform mat, as in SFMlearner
    batch_size = R_left.shape[0]
    filler = torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(device).reshape(1, 1, 4)  # [1, 1, 4]
    filler = filler.repeat(batch_size, 1, 1)  # [B, 1, 4]
    transform_mat = torch.cat([R_rel, t_rel], dim=2)  # [B, 3, 4]
    transform_mat = torch.cat([transform_mat.float(), filler.float()], dim=1)  # [B, 4, 4]

    batch_size, img_height, img_width, _ = img.shape
    # print('depth: {}'.format(depth.shape))
    depth = depth.reshape(batch_size, 1, img_height * img_width)  # [batch_size, 1, height * width]

    grid = _meshgrid_abs(img_height, img_width)  # [3, height * width]
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 3, height * width]
    cam_coords = _pixel2cam(depth, grid, K_left_inv)  # [batch_size, 3, height * width]
    ones = torch.ones([batch_size, 1, img_height * img_width], device=device)  # [batch_size, 1, height * width]
    cam_coords_hom = torch.cat([cam_coords, ones], dim=1)  # [batch_size, 4, height * width]

    # Get projection matrix for target camera frame to source pixel frame
    hom_filler = torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(device).reshape(1, 1, 4)  # [1, 1, 4]
    hom_filler = hom_filler.repeat(batch_size, 1, 1)  # [B, 1, 4]
    intrinsic_mat_hom = torch.cat([K_left.float(), torch.zeros([batch_size, 3, 1], device=device)], dim=2)  # [B, 3, 4]
    intrinsic_mat_hom = torch.cat([intrinsic_mat_hom, hom_filler], dim=1)  # [B, 4, 4]
    proj_target_cam_to_source_pixel = torch.matmul(intrinsic_mat_hom, transform_mat)  # [B, 4, 4]
    source_pixel_coords = _cam2pixel(cam_coords_hom, proj_target_cam_to_source_pixel)  # [batch_size, 2, height * width]
    source_pixel_coords = source_pixel_coords.reshape(batch_size, 2, img_height, img_width)   # [batch_size, 2, height, width]
    source_pixel_coords = source_pixel_coords.permute(0, 2, 3, 1)  # [batch_size, height, width, 2]
    warped_right, mask = _spatial_transformer(img, source_pixel_coords)
    return warped_right, mask


def _meshgrid_abs(height, width):
    """Meshgrid in the absolute coordinates."""
    x_t = torch.matmul(
        torch.ones([height, 1]),
        torch.linspace(-1.0, 1.0, width).unsqueeze(1).permute(1, 0)
    )  # [height, width]
    y_t = torch.matmul(
        torch.linspace(-1.0, 1.0, height).unsqueeze(1),
        torch.ones([1, width])
    )
    x_t = (x_t + 1.0) * 0.5 * (width - 1)
    y_t = (y_t + 1.0) * 0.5 * (height - 1)
    x_t_flat = x_t.reshape(1, -1)
    y_t_flat = y_t.reshape(1, -1)
    ones = torch.ones_like(x_t_flat)
    grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)  # [3, height * width]
    return grid.to(device)


def _pixel2cam(depth, pixel_coords, intrinsic_mat_inv):
    """Transform coordinates in the pixel frame to the camera frame."""
    cam_coords = torch.matmul(intrinsic_mat_inv.float(), pixel_coords.float()) * depth.float()
    return cam_coords


def _cam2pixel(cam_coords, proj_c2p):
    """Transform coordinates in the camera frame to the pixel frame."""
    pcoords = torch.matmul(proj_c2p, cam_coords)  # [batch_size, 4, height * width]
    x = pcoords[:, 0:1, :]  # [batch_size, 1, height * width]
    y = pcoords[:, 1:2, :]  # [batch_size, 1, height * width]
    z = pcoords[:, 2:3, :]  # [batch_size, 1, height * width]
    x_norm = x / (z + 1e-10)
    y_norm = y / (z + 1e-10)
    pixel_coords = torch.cat([x_norm, y_norm], dim=1)
    return pixel_coords  # [batch_size, 2, height * width]


def _spatial_transformer(img, coords):
    """A wrapper over binlinear_sampler(), taking absolute coords as input."""
    # img: [B, H, W, C]
    img_height = img.shape[1]
    img_width = img.shape[2]
    px = coords[:, :, :, :1]  # [batch_size, height, width, 1]
    py = coords[:, :, :, 1:]  # [batch_size, height, width, 1]
    # Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
    px = px / (img_width - 1) * 2.0 - 1.0  # [batch_size, height, width, 1]
    py = py / (img_height - 1) * 2.0 - 1.0  # [batch_size, height, width, 1]
    output_img, mask = _bilinear_sample(img, px, py)
    return output_img, mask


def _bilinear_sample(im, x, y, name='bilinear_sampler'):
    """Perform bilinear sampling on im given list of x, y coordinates.
    Implements the differentiable sampling mechanism with bilinear kernel
    in https://arxiv.org/abs/1506.02025.
    x,y are tensors specifying normalized coordinates [-1, 1] to be sampled on im.
    For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
    and (1, 1) in (x, y) corresponds to the bottom right pixel in im.
    Args:
        im: Batch of images with shape [B, h, w, channels].
        x: Tensor of normalized x coordinates in [-1, 1], with shape [B, h, w, 1].
        y: Tensor of normalized y coordinates in [-1, 1], with shape [B, h, w, 1].
        name: Name scope for ops.
    Returns:
        Sampled image with shape [B, h, w, channels].
        Principled mask with shape [B, h, w, 1], dtype:float32.  A value of 1.0
        in the mask indicates that the corresponding coordinate in the sampled
        image is valid.
      """
    x = x.reshape(-1)  # [batch_size * height * width]
    y = y.reshape(-1)  # [batch_size * height * width]

    # Constants.
    batch_size, height, width, channels = im.shape

    x, y = x.float(), y.float()
    max_y = int(height - 1)
    max_x = int(width - 1)

    # Scale indices from [-1, 1] to [0, width - 1] or [0, height - 1].
    x = (x + 1.0) * (width - 1.0) / 2.0
    y = (y + 1.0) * (height - 1.0) / 2.0

    # Compute the coordinates of the 4 pixels to sample from.
    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    mask = (x0 >= 0) & (x1 <= max_x) & (y0 >= 0) & (y0 <= max_y)
    mask = mask.float()

    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)
    y0 = torch.clamp(y0, 0, max_y)
    y1 = torch.clamp(y1, 0, max_y)
    dim2 = width
    dim1 = width * height

    # Create base index.
    base = torch.arange(batch_size) * dim1
    base = base.reshape(-1, 1)
    base = base.repeat(1, height * width)
    base = base.reshape(-1)  # [batch_size * height * width]
    base = base.long().to(device)

    base_y0 = base + y0.long() * dim2
    base_y1 = base + y1.long() * dim2
    idx_a = base_y0 + x0.long()
    idx_b = base_y1 + x0.long()
    idx_c = base_y0 + x1.long()
    idx_d = base_y1 + x1.long()

    # Use indices to lookup pixels in the flat image and restore channels dim.
    im_flat = im.reshape(-1, channels).float()  # [batch_size * height * width, channels]
    # pixel_a = tf.gather(im_flat, idx_a)
    # pixel_b = tf.gather(im_flat, idx_b)
    # pixel_c = tf.gather(im_flat, idx_c)
    # pixel_d = tf.gather(im_flat, idx_d)
    pixel_a = im_flat[idx_a]
    pixel_b = im_flat[idx_b]
    pixel_c = im_flat[idx_c]
    pixel_d = im_flat[idx_d]

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (1.0 - (y1.float() - y))
    wc = (1.0 - (x1.float() - x)) * (y1.float() - y)
    wd = (1.0 - (x1.float() - x)) * (1.0 - (y1.float() - y))
    wa, wb, wc, wd = wa.unsqueeze(1), wb.unsqueeze(1), wc.unsqueeze(1), wd.unsqueeze(1)

    output = wa * pixel_a + wb * pixel_b + wc * pixel_c + wd * pixel_d
    output = output.reshape(batch_size, height, width, channels)
    mask = mask.reshape(batch_size, height, width, 1)
    return output, mask


if __name__ == '__main__':
    from datasets.dtu_yao2 import MVSDataset

    view_num = 3
    depth_num = 192
    depth_interval = 1.06
    datapath = "D:\\BaiduNetdiskDownload\\mvsnet\\training_data\\dtu_training"
    listfile = "E:\\PycharmProjects\\un_mvsnet_pytorch\\lists\\dtu\\train.txt"
    train_dataset = MVSDataset(datapath, listfile, "train", nviews=view_num, ndepths=depth_num, interval_scale=depth_interval)
    print('dataset length: {}'.format(len(train_dataset)))
    item = train_dataset[100]
    print(item.keys())
    print("imgs", item["imgs"].shape)
    print("depth", item["depth"].shape)
    print("cams", item["cams"].shape)

    depth_start = item["depth_start"]
    depth_end = depth_start + (depth_num - 1) * depth_interval

    # reference image
    ref_image = torch.tensor(item["imgs"][0]).unsqueeze(0)  # [B, C, H, W]
    ref_image = F.interpolate(ref_image, scale_factor=0.25, mode='bilinear')
    ref_image = ref_image.permute(0, 2, 3, 1)
    ref_cam = torch.tensor(item["cams"][0]).unsqueeze(0)  # [B, 2, 4, 4]
    # depth
    depth = torch.tensor(item["depth"]).unsqueeze(0).squeeze(-1)

    # get all homographies
    warped_images_np = []
    view_images_np = []
    masks_np = []
    for view in range(1, view_num):
        view_cam = torch.tensor(item["cams"][view]).unsqueeze(0)  # [B, 2, 4, 4]
        view_image = torch.tensor(item["imgs"][view]).unsqueeze(0)  # [B, C, H, W]
        view_image = F.interpolate(view_image, scale_factor=0.25, mode='bilinear')
        view_image = view_image.permute(0, 2, 3, 1)

        warped_img, mask = inverse_warping(view_image, ref_cam, view_cam, depth)

        warped_images_np.append(warped_img[0].cpu().numpy())
        view_images_np.append(view_image[0].cpu().numpy())
        masks_np.append(mask.repeat(1, 1, 1, 3)[0].cpu().numpy())

    ref_image_np = ref_image[0].cpu().numpy()

    from matplotlib import pyplot as plt
    import cv2
    plt.figure(figsize=[12, 8])
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(ref_image_np, cv2.COLOR_BGR2RGB))
    plt.title('ref')
    plt.subplot(3, 3, 2)
    plt.imshow(cv2.cvtColor(view_images_np[0], cv2.COLOR_BGR2RGB))
    plt.title('src1')
    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(view_images_np[1], cv2.COLOR_BGR2RGB))
    plt.title('src2')
    plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(warped_images_np[0], cv2.COLOR_BGR2RGB) * masks_np[0])
    # plt.imshow(warped_images_np[0])
    plt.title('src1-->ref warp')
    plt.subplot(3, 3, 6)
    plt.imshow(cv2.cvtColor(warped_images_np[1], cv2.COLOR_BGR2RGB) * masks_np[1])
    # plt.imshow(warped_images_np[1])
    plt.title('src2-->ref warp')
    plt.subplot(3, 3, 8)
    plt.imshow(masks_np[0])
    plt.title('src1-->ref mask')
    plt.subplot(3, 3, 9)
    plt.imshow(masks_np[1])
    plt.title('src2-->ref mask')
    plt.tight_layout()
    plt.show()





