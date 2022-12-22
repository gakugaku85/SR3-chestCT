from skimage import metrics
import os
import numpy as np
import SimpleITK as sitk
import math
import cv2
from natsort import natsorted

import pandas as pd


def ssim(img1, img2):
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
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
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
        raise ValueError("Wrong input image dimensions.")


def psnr_new(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))



def psnr_ssim(input_path, gt_path):
    im_1 = sitk.ReadImage(input_path)
    im_1 = sitk.GetArrayFromImage(im_1)

    im_2 = sitk.ReadImage(gt_path)
    im_2 = sitk.GetArrayFromImage(im_2)

    im_1 = im_1.astype(np.float64)
    im_2 = im_2.astype(np.float64)
    psnr = psnr_new(im_1, im_2)
    ssim = calculate_ssim(im_1, im_2)

    return psnr, ssim


def save_scores(result_list):

    df = pd.DataFrame(result_list, columns=['val_index', 'PSNR_proposed_iter4', 'SSIM_proposed_iter4'])
    # df = df.round({'Dice': 3})
    df.to_csv(r"\\take.lab\lung\samarth\DAN-master_plus_dwdn\experiments\DANv1_plus_dwdn_GT_kernels\train_setting1_x4_lr_reduced_bic_val_GT_kernels\iter115000_GT_no_sig2point4.csv", index=False)

def main():
    import argparse

    prog = argparse.ArgumentParser()
    prog.add_argument('--proposed_path', '-i', type=str,
                      default=r"\\take.lab\lung\samarth\DAN_plus_dwdn_GT_motion_blur\codes_16bit_kernels_flipped_data_only_lung\experiments\GT_sameasabove_ZERO_PAD\GT_HRsize64_onlylung_ZERO_PAD\ValResults_1792_NEW_DENOISED_small\Val_results",
                      help='path to input image directory')

    prog.add_argument('--gt_path', '-o', type=str,
                      default=r"\\take.lab\lung\samarth\DAN_plus_dwdn_GT_motion_blur\codes_16bit_kernels_flipped_data_only_lung\data\ValData_SR_kernelsflipped_maxsize_37_BIC_DS\1792_new_denoised\Resized_HR_LR\HR_small",
                      help='output directory path')

    args = prog.parse_args()

    input_image_paths_proposed = []
    gt_paths = []

    for l in range(2500, 115000, 2500):
        args.final_proposed_path = os.path.join(args.proposed_path, str(l) + '_all_models')

        for img_filename in natsorted(os.listdir(os.path.abspath(args.final_proposed_path))):
            if img_filename.endswith('.mhd'):
                input_image_paths_proposed.append(img_filename)

        for mask_filename in natsorted(os.listdir(os.path.abspath(args.gt_path))):
            if mask_filename.endswith('.mhd'):
                gt_paths.append(mask_filename)

        psnr_list_proposed, ssim_list_proposed = [], []
        results = []
        for i in range(len(input_image_paths_proposed)):
            input_image_path_proposed = os.path.join(args.final_proposed_path, input_image_paths_proposed[i])
            gt_path = os.path.join(args.gt_path, gt_paths[i])
            # print('Input path_proposed,  Input path_DAN, output path =', input_image_path_proposed, gt_path)

            psnr_new, ssim_new = psnr_ssim(input_image_path_proposed, gt_path)
            # print(l, psnr_new, ssim_new)

            psnr_list_proposed.append(psnr_new)
            ssim_list_proposed.append(ssim_new)

            results.append([i, psnr_new, ssim_new])

        psnr_avg_new = np.sum(psnr_list_proposed)/len(psnr_list_proposed)
        ssim_avg_new = np.sum(ssim_list_proposed)/len(ssim_list_proposed)

        print('model no., psnr_new, ssim_new = ', l, psnr_avg_new, ssim_avg_new)

        # save_scores(results)

        # prog.exit(0)


if __name__ == '__main__':
    main()
