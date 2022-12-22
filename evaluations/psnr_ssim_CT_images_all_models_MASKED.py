from skimage import metrics
import os
import numpy as np
import SimpleITK as sitk
import math
import cv2
from natsort import natsorted
import PIL.Image as Image

import pandas as pd


def ssim(img1, img2, mask):
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


def calculate_ssim(img1, img2, mask):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2, mask)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2, mask))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2), mask)
    else:
        raise ValueError("Wrong input image dimensions.")


def psnr_new(img1, img2, mask):
    square_errors = (img1 - img2) ** 2
    masked_square_errors = np.ma.array(square_errors, mask=mask)
    masked_mse = np.mean(masked_square_errors)
    if masked_mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(masked_mse))



def psnr_ssim(input_path, gt_path, mask_path):
    im_1 = sitk.ReadImage(input_path)
    im_1 = sitk.GetArrayFromImage(im_1)

    msk = Image.open(mask_path).convert('L')
    msk = np.array(msk, dtype=np.uint8)
    # print(msk.shape)
    #Resize mask images, to fit with resized gt. For 1792
    # msk = msk[:1712, :1936]

    mask_slice = msk / 255
    mask_array_boolean = np.ma.make_mask(mask_slice)
    mask_array = np.invert(mask_array_boolean)
    # Convert back to uint8 from float
    # mask_slice = mask_slice.astype(np.uint8)
    # print(im_1.shape, mask_slice.shape)

    masked_im1 = im_1 * mask_slice
    im_2 = sitk.ReadImage(gt_path)
    im_2 = sitk.GetArrayFromImage(im_2)

    # print(im_1.shape, masked_im1.shape, im_2.shape, mask_array.shape)

    masked_im1 = masked_im1.astype(np.float64)
    im_2 = im_2.astype(np.float64)

    psnr = psnr_new(masked_im1, im_2, mask_array)
    ssim = calculate_ssim(masked_im1, im_2, mask_array)

    return psnr, ssim


def save_scores(result_list):

    df = pd.DataFrame(result_list, columns=['val_index', 'PSNR_proposed_iter4', 'SSIM_proposed_iter4'])
    # df = df.round({'Dice': 3})
    df.to_csv(r"\\psnr.csv", index=False)

def main():
    import argparse

    prog = argparse.ArgumentParser()
    prog.add_argument('--proposed_path', '-i', type=str,
                      default=r"",
                      help='path to input image directory')

    prog.add_argument('--gt_path', '-o', type=str,
                      default=r"",
                      help='output directory path')
    prog.add_argument('--masks_path', '-k', type=str,
                      default=r"",
                      help='output directory path')
    args = prog.parse_args()

    results = []

    for l in range(170000, 240000, 2500):
        input_image_paths_proposed = []
        gt_paths = []
        mask_paths = []

        args.final_proposed_path = os.path.join(args.proposed_path, str(l) + '_all_models')

        for img_filename in natsorted(os.listdir(os.path.abspath(args.final_proposed_path))):
            if img_filename.endswith('.mhd'):
                input_image_paths_proposed.append(img_filename)

        for mask_filename in natsorted(os.listdir(os.path.abspath(args.gt_path))):
            if mask_filename.endswith('.mhd'):
                gt_paths.append(mask_filename)

        for mask_filename in natsorted(os.listdir(os.path.abspath(args.masks_path))):
            mask_paths.append(mask_filename)

        psnr_list_proposed, ssim_list_proposed = [], []
        print(len(input_image_paths_proposed))
        for i in range(len(input_image_paths_proposed)):
            print(i)
            input_image_path_proposed = os.path.join(args.final_proposed_path, input_image_paths_proposed[i])
            gt_path = os.path.join(args.gt_path, gt_paths[i])
            mask_path = os.path.join(args.masks_path, mask_paths[i])

            print('Input path, mask path, output path =', input_image_path_proposed, mask_path, gt_path)

            psnr_new, ssim_new = psnr_ssim(input_image_path_proposed, gt_path, mask_path)
            # print(l, psnr_new, ssim_new)

            psnr_list_proposed.append(psnr_new)
            ssim_list_proposed.append(ssim_new)

            # results.append([i, psnr_new, ssim_new])

        psnr_avg_new = np.sum(psnr_list_proposed)/len(psnr_list_proposed)
        ssim_avg_new = np.sum(ssim_list_proposed)/len(ssim_list_proposed)
        results.append([l, psnr_avg_new, ssim_avg_new])

        print('model no., psnr_new, ssim_new = ', l, psnr_avg_new, ssim_avg_new)

    save_scores(results)

        # prog.exit(0)


if __name__ == '__main__':
    main()
