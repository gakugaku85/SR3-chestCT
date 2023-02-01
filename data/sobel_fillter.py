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
                        default='../dataset/microCT_slices_1794_0_0_2')
    parser.add_argument('--out', '-o', type=str,
                        default='../dataset/mask_1794_sobel_png_1_reverse')

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float64)

    files = natsorted(glob(osp.join(args.path+'/hr_0/*.mhd'))) #スライスファイルの名前取得
    # print(files)

    val_files = []
    for i, path in tqdm(enumerate(files)):
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))
        image = sobel_filter(image)
        # img_bin = np.where((image > 0.5) & (image < 1), 255, 0)
        img_bin = np.where(image < 1, 255, 0)
        save_img(img_bin, '{}/{}_hr_bin1.png'.format(args.out, i))
        # save_mhd(image, '{}/{}_hr.mhd'.format(args.out, i))