import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
import SimpleITK as sitk


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def tensor2mhd(tensor, out_type=np.float64, min_max=(0, 1)):  # type: ignore
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.float64:
        img_np = img_np * 255.
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)

def save_mhd(img, img_path):
    # if img.ndim == 3:
    #     img = img[:, :, 0]
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, img_path)

def concatImage(images, opt):
    h = opt['datasets']['val']['image_h']
    w = opt['datasets']['val']['image_w']
    hr_patch_size = opt['datasets']['val']['r_resolution']
    overlap = opt['datasets']['val']['overlap']
    STRIDE = hr_patch_size - overlap #60
    coor = [(x, y)
            for x in range(0, w, STRIDE)
            for y in range(0, h, STRIDE)]
    overlap_im = np.zeros((w+hr_patch_size, h+hr_patch_size), dtype=np.float64)
    count = np.zeros(overlap_im.shape, dtype=np.uint8)

    for i, (x, y) in enumerate(coor):
        overlap_im[x:x+hr_patch_size, y:y+hr_patch_size] += images[i]
        count[x:x+hr_patch_size, y:y+hr_patch_size] += 1
    overlap_im[count > 0] /= count[count > 0]
    return overlap_im[:w, :h]

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_psnr_mask(img1, img2, mask):
    mask_slice = mask / 255
    mask_array_boolean = np.ma.make_mask(mask_slice)
    mask_array = np.invert(mask_array_boolean)

    masked_im1 = img1 * mask_slice
    masked_im1 = masked_im1.astype(np.float64)

    square_errors = (masked_im1 - img2) ** 2
    masked_square_errors = np.ma.array(square_errors, mask=mask_array)
    masked_mse = np.mean(masked_square_errors)
    if masked_mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(masked_mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def ssim_mask(img1, img2, mask):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    ssim_map_masked = np.ma.array(ssim_map, mask=mask[5:-5, 5:-5])
    return ssim_map_masked.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_ssim_mask(img1, img2, mask):
    mask_slice = mask / 255
    mask_array_boolean = np.ma.make_mask(mask_slice)
    mask_array = np.invert(mask_array_boolean)

    masked_im1 = img1 * mask_slice
    masked_im1 = masked_im1.astype(np.float64)
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim_mask(masked_im1, img2, mask_array)
    else:
        raise ValueError("Wrong input image dimensions.")

def zncc(img1, img2):
    # 画像をfloat64型に変換
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # 画像の平均値を計算
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    # 画像を平均値0に正規化
    img1 = img1 - mean1
    img2 = img2 - mean2
    # 相関係数を計算
    corr = np.sum(img1 * img2) / (np.sqrt(np.sum(img1 ** 2)) * np.sqrt(np.sum(img2 ** 2)))
    return corr

def d_power(img1, img2):
    # 画像をfloat64型に変換
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # 画像を高速フーリエ変換
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    # 高周波成分の強さを計算
    high_pass1 = np.abs(f1) ** 2
    high_pass2 = np.abs(f2) ** 2
    # D-powerを計算
    dp = np.sum(high_pass2) / np.sum(high_pass1)
    return dp