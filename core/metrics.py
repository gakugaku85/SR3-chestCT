import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
import SimpleITK as sitk
from scipy.fftpack import fftn


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
    # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # img = np.uint8(img)
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(img_path, img)

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
            for x in range(0, h, STRIDE)
            for y in range(0, w, STRIDE)]
    overlap_im = np.zeros((h+hr_patch_size, w+hr_patch_size), dtype=np.float64)
    count = np.zeros(overlap_im.shape, dtype=np.uint8)

    for i, (x, y) in enumerate(coor):
        overlap_im[x:x+hr_patch_size, y:y+hr_patch_size] += images[i]
        count[x:x+hr_patch_size, y:y+hr_patch_size] += 1
    overlap_im[count > 0] /= count[count > 0]
    return overlap_im[:h, :w]

def calculate_psnr(gt, input):
    # gt and input have range [0, 255]
    gt = gt.astype(np.float64)
    input = input.astype(np.float64)
    mse = np.mean((gt - input)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_psnr_mask(gt, input, mask):
    mask_slice = mask / 255
    mask_array_boolean = np.ma.make_mask(mask_slice)
    mask_array = np.invert(mask_array_boolean)

    masked_im1 = gt * mask_slice
    masked_im1 = masked_im1.astype(np.float64)

    square_errors = (masked_im1 - input) ** 2
    masked_square_errors = np.ma.array(square_errors, mask=mask_array)
    masked_mse = np.mean(masked_square_errors)
    if masked_mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(masked_mse))


def ssim(gt, input):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    gt = gt.astype(np.float64)
    input = input.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(gt, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(input, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(gt**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(input**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(gt * input, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def ssim_mask(gt, input, mask):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    gt = gt.astype(np.float64)
    input = input.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(gt, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(input, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(gt ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(input ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(gt * input, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    ssim_map_masked = np.ma.array(ssim_map, mask=mask[5:-5, 5:-5])
    return ssim_map_masked.mean()


def calculate_ssim(gt, input):
    '''calculate SSIM
    the same outputs as MATLAB's
    gt, input: [0, 255]
    '''
    if not gt.shape == input.shape:
        raise ValueError('Input images must have the same dimensions.')
    if gt.ndim == 2:
        return ssim(gt, input)
    elif gt.ndim == 3:
        if gt.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(gt, input))
            return np.array(ssims).mean()
        elif gt.shape[2] == 1:
            return ssim(np.squeeze(gt), np.squeeze(input))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_ssim_mask(gt, input, mask):
    mask_slice = mask / 255
    mask_array_boolean = np.ma.make_mask(mask_slice)
    mask_array = np.invert(mask_array_boolean)

    masked_im1 = gt * mask_slice
    masked_im1 = masked_im1.astype(np.float64)
    """calculate SSIM
    the same outputs as MATLAB's
    gt, input: [0, 255]
    """
    if not gt.shape == input.shape:
        raise ValueError("Input images must have the same dimensions.")
    if gt.ndim == 2:
        return ssim_mask(masked_im1, input, mask_array)
    else:
        raise ValueError("Wrong input image dimensions.")

def zncc(gt, input):
    # 画像をfloat64型に変換
    gt = gt.astype(np.float64)
    input = input.astype(np.float64)
    # 画像の平均値を計算
    mean1 = np.mean(gt)
    mean2 = np.mean(input)
    # 画像を平均値0に正規化
    gt = gt - mean1
    input = input - mean2
    # 相関係数を計算
    corr = np.sum(gt * input) / (np.sqrt(np.sum(gt ** 2)) * np.sqrt(np.sum(input ** 2)))
    return corr

def calc_zncc(gt, input):
    """
    https://github.com/ladisk/pyDIC/blob/master/dic.py
    Calculate the zero normalized cross-correlation coefficient of input images.
    :param gt: First input image.
    :param input: Second input image.
    :return: zncc ([0,1]). If 1, input images match perfectly.
    """
    nom = np.mean((gt-gt.mean())*(input-input.mean()))
    den = gt.std()*input.std()
    if den == 0:
        return 0
    return nom/den

def d_power(gt, input):
    # 画像をfloat64型に変換
    gt = gt.astype(np.float64)
    input = input.astype(np.float64)
    # 画像を高速フーリエ変換
    f1 = np.fft.fft2(gt)
    f2 = np.fft.fft2(input)
    # 高周波成分の強さを計算
    high_pass1 = np.abs(f1) ** 2
    high_pass2 = np.abs(f2) ** 2
    # D-powerを計算
    dp = np.sum(high_pass2) / np.sum(high_pass1)
    return dp

def calc_fft_domain(gt, input):
    imgFreqs = np.fft.fftn(input)
    gtFreqs = np.fft.fftn(gt)
    imgFreqs = np.fft.fftshift(imgFreqs)
    gtFreqs = np.fft.fftshift(gtFreqs)

    img_freq = np.abs(imgFreqs)
    gt_freq = np.abs(gtFreqs)

    diff_spe = np.abs((gt_freq**2)-(img_freq)**2)
    diff_spe_const = np.mean(diff_spe)

    return diff_spe_const, diff_spe, img_freq, gt_freq

def mean_values_by_distance(image, num_pixels):
    # 画像の高さと幅
    height, width = image.shape

    center_x = width // 2
    center_y = height // 2
    values = []
    for i in range(0, num_pixels):
        sum_values = 0
        pixels = 0
        for y in range(height):
            for x in range(width):
                distance = np.maximum(np.abs(x - center_x), np.abs(y - center_y))
                if distance == i:
                    sum_values += image[y, x].mean()
                    pixels += 1
        if pixels == 0:
            mean_value = 0
        else:
            mean_value = sum_values / pixels
        values.append(mean_value)

    return values

# def mean_value_new(img, num_pixels):
#     height, width = img.shape

#     center_x = width // 2
#     center_y = height // 2

#     value = []

#     for i in range(height):