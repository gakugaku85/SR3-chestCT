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


def concatImage(images, h, w, hr_size):
    H = h
    W = w
    while H % hr_size != 0:
        H=H+1
    while W % hr_size != 0:
        W=W+1
    nH = int(H/hr_size)
    nW = int(W/hr_size)
    image_h = []
    for i in range(nW):
        image_h.append(np.concatenate(images[i*nH:(i+1)*nH], axis=3))
    image = np.concatenate(image_h, axis=2)
    print(image.shape)
    image = image[:, :, :w, :h]
    return image

def save_mhd(img, img_path):
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, img_path)

def sobel_filter(image, min_max=(0, 1)):
    img = np.array(image)
    img = img.astype(np.float64) / 255.
    img = img.clip(min=0, max=1)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    image = img*(min_max[1] - min_max[0]) + min_max[0]

    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float64)

    conv_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, bias=False)
    conv_x.weight = nn.Parameter(kernel_x.unsqueeze(0).unsqueeze(0))

    conv_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, bias=False)
    conv_y.weight = nn.Parameter(kernel_y.unsqueeze(0).unsqueeze(0))

    image = image.unsqueeze(0)

    output_x = conv_x(image)
    output_y = conv_y(image)

    output = torch.abs(output_x) + torch.abs(output_y)
    # output = torch.sqrt(torch.pow(output_x, 2) + torch.pow(output_y, 2))
    output = output.detach().numpy()

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='../dataset/microCT_slices_1794_patch_16_64_2')
    parser.add_argument('--out', '-o', type=str,
                        default='../dataset/microCT_slices_1794_patch_16_64_sobel')

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float64)

    files = glob(osp.join(args.path +'/hr_64/*')) #スライスファイルの名前取得
    files = natsorted([file.split('/')[4] for file in files])

    val_files = []
    for i, file in enumerate(files):
        patch_paths = Util.get_paths_from_mhds('{}/hr_{}/{}'.format(args.path, 64, file))
        patch_images = []
        for path in tqdm(patch_paths):
            patch_image = sitk.GetArrayFromImage(sitk.ReadImage(path))
            patch_image = sobel_filter(patch_image)
            patch_images.append(patch_image)
        val_files.append(patch_images)
        hr_sobel = concatImage(patch_images, 1100, 1536, 64)
        save_mhd(hr_sobel[0][0], '{}/{}_hr.mhd'.format(args.out, i+3))