import SimpleITK as sitk
import argparse
from glob import glob
import numpy as np
import util as Util
from natsort import natsorted
import os
import os.path as osp
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2
from scipy import ndimage

def save_mhd(img, img_path):
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, img_path)

def save_img(img, img_path, mode='RGB'):
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(img_path, img)

def sobel_filter(image, min_max=(0, 1)):
    img = np.array(image)
    img = img.astype(np.float64) / 255.
    img = img.clip(min=0, max=1)

    # Define Sobel filters
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply Sobel filters
    sobelx = ndimage.convolve(img, sobel_x)
    sobely = ndimage.convolve(img, sobel_y)
    sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    # sobel = np.abs(sobelx) + np.abs(sobely)

    return sobel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='../dataset/microCT_slices_1792_0_0_2')
    parser.add_argument('--mask_path', '-mp', type=str,
                        default='../dataset/mask_1792')
    parser.add_argument('--out', '-o', type=str,
                        default='../dataset/mask_1792_sobel_png')

    args = parser.parse_args()
    os.makedirs(args.out+'_1', exist_ok=True)
    os.makedirs(args.out+'_2', exist_ok=True)
    os.makedirs(args.out+'_3', exist_ok=True)
    os.makedirs(args.out+'_05', exist_ok=True)
    os.makedirs(args.out+'_E', exist_ok=True)
    os.makedirs(args.out+'_1_outside', exist_ok=True)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float64)

    mask_files = natsorted(glob(osp.join(args.mask_path+'_png_2/*.png')))
    files = natsorted(glob(osp.join(args.path+'/hr_0/*.mhd'))) #スライスファイルの名前取得
    i = 0
    val_files = []
    for path, mask in zip(files, mask_files):
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))
        mask = cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2GRAY)
        image = sobel_filter(image)
        print(np.max(image), np.min(image))
        img_bin_E = np.where((image > 0.5) & (image < 1), 255, 0)
        img_bin_1 = np.where(image > 1, 255, 0)
        img_bin_2 = np.where(image > 2, 255, 0)
        img_bin_3 = np.where(image > 3, 255, 0)
        img_bin_05 = np.where(image > 0.5, 255, 0)

        img_bin_rev = np.where(image < 1, 255, 0)
        img_outside = mask * img_bin_rev

        save_img(img_bin_1, '{}_1/{}_hr_bin1.png'.format(args.out, i))
        save_img(img_bin_2, '{}_2/{}_hr_bin2.png'.format(args.out, i))
        save_img(img_bin_3, '{}_3/{}_hr_bin3.png'.format(args.out, i))
        save_img(img_bin_05, '{}_05/{}_hr_bin1.png'.format(args.out, i))
        save_img(img_bin_E, '{}_E/{}_hr_bin1.png'.format(args.out, i))
        save_img(img_outside, '{}_1_outside/{}_hr_bin1.png'.format(args.out, i))
        # save_mhd(image, '{}/{}_hr.mhd'.format(args.out, i))
        i += 1