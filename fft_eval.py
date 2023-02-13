import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
from glob import glob
import os
import SimpleITK as sitk
import os.path as osp
from natsort import natsorted
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments/sr_sam_dan/my_results')
    args = parser.parse_args()

    mask_root_path = "../dataset/mask_1792"
    mask_list_path = ['_png_2/*.png', '_sobel_png_1/*.png', '_sobel_png_1_outside/*.png', '_sobel_png_E/*.png']
    
    mask_files = [natsorted(glob(osp.join(mask_root_path+path))) for path in mask_list_path]

    mask_list = []
    for mask_file in mask_files:
        mask_imgs = []
        for mask_path in mask_file:
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask, dtype=np.uint8)
            mask_imgs.append(mask)
        mask_list.append(mask_imgs)

    test_paths =['experiments/sr_microCT_patch_val_230206_012720/results',
                'experiments/sr_microCT_patch_val_230206_012438/results',
                'experiments/sr_sam_dan/my_results']

    result_path = './experiments/eval_fft/'
    result_path1 = result_path + 'result'
    result_path2 = result_path + 'spect'
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(result_path1, exist_ok=True)
    os.makedirs(result_path2, exist_ok=True)

    comp_hr_names, comp_sr_names, comp_lr_names = [], [], []

    ipx = 0
    for test_path in test_paths:
        hr_names = natsorted(glob(osp.join('{}/*_hr.mhd'.format(test_path))))
        sr_names = natsorted(glob(osp.join('{}/*_sr.mhd'.format(test_path))))
        lr_names = natsorted(glob(osp.join('{}/*_inf.mhd'.format(test_path))))

        comp_hr_names.append(hr_names)
        comp_sr_names.append(sr_names)
        comp_lr_names.append(lr_names)
        ipx+=1

    for img_i in range(len(hr_names)):
        for path_i in range(len(comp_hr_names)):
            img_HR = sitk.GetArrayFromImage(sitk.ReadImage(comp_hr_names[path_i][img_i]))
            img_SR = sitk.GetArrayFromImage(sitk.ReadImage(comp_sr_names[path_i][img_i]))
            img_LR = sitk.GetArrayFromImage(sitk.ReadImage(comp_lr_names[path_i][img_i]))
            h, w = img_HR.shape
            center_y = h // 2
            center_x = w // 2
            size = 256
            hr_org = img_HR[center_y-size:center_y+size, center_x-size:center_x+size].astype(np.float32)
            sr_org = img_SR[center_y-size:center_y+size, center_x-size:center_x+size].astype(np.float32)
            lr_org = img_LR[center_y-size:center_y+size, center_x-size:center_x+size].astype(np.float32)
            # hr_org = img_HR.astype(np.float32)
            # sr_org = img_SR.astype(np.float32)
            # lr_org = img_LR.astype(np.float32)

            diff_spe_const, diff_spe, sr_spe, hr_spe = Metrics.calc_fft_domain(hr_org, sr_org)

            Metrics.save_mhd(hr_org, '{}/0_{}hr_org.mhd'.format(result_path2, img_i))
            Metrics.save_mhd(lr_org, '{}/0_{}lr_org.mhd'.format(result_path2, img_i))
            Metrics.save_mhd(sr_org, '{}/{}_{}sr_org.mhd'.format(result_path2, path_i, img_i))

            hr_fft = (hr_org - hr_org.min()) / (hr_org.max() - hr_org.min())
            sr_fft = (sr_org - sr_org.min()) / (sr_org.max() - sr_org.min())
            lr_fft = (lr_org - lr_org.min()) / (lr_org.max() - lr_org.min())
            hr_fft = hr_fft * 2 - 1
            sr_fft = sr_fft * 2 - 1
            lr_fft = lr_fft * 2 - 1

            pow_hr = np.abs(np.fft.fftshift(np.fft.fftn(hr_fft)))**2
            pow_sr = np.abs(np.fft.fftshift(np.fft.fftn(sr_fft)))**2
            pow_lr = np.abs(np.fft.fftshift(np.fft.fftn(lr_fft)))**2

            norm_pow_spe_hr = 10 * np.log(pow_hr)
            norm_pow_spe_sr = 10 * np.log(pow_sr)
            norm_pow_spe_lr = 10 * np.log(pow_lr)

            if path_i == 0:
                freq_hr = Metrics.mean_values_by_distance(norm_pow_spe_hr, size)
                freq_lr = Metrics.mean_values_by_distance(norm_pow_spe_lr, size)
            freq_sr = Metrics.mean_values_by_distance(norm_pow_spe_sr, size)

            # if path_i == 0:
            #     freq_hr = Metrics.mean_values_by_distance(pow_hr, size)
            #     freq_lr = Metrics.mean_values_by_distance(pow_lr, size)
            # freq_sr = Metrics.mean_values_by_distance(pow_sr, size)

            # freq_sub_srhr = [np.abs(a - b) for a, b in zip(freq_hr, freq_sr)]

            if path_i == 0:
                plt.plot(freq_hr, label='hr_spect', linewidth=1)
                plt.plot(freq_lr, label='lr_spect', linewidth=1)
            plt.plot(freq_sr, label='sr_spect{}'.format(path_i+1), linewidth=1)
            # plt.plot(freq_sub_srhr, label='sr_spect{}'.format(path_i+1), linewidth=1)

            def array2sitk(arr, spacing=[], origin=[]):
                    if not len(spacing) == arr.ndim and len(origin) == arr.ndim:
                        print("Dimension Error")
                        quit()
                    sitkImg = sitk.GetImageFromArray(arr)
                    sitkImg.SetSpacing(spacing)
                    sitkImg.SetOrigin(origin)
                    return sitkImg

            pow_img_hr = array2sitk(pow_hr, [1,1], [0,0])
            pow_img_sr = array2sitk(pow_sr, [1,1], [0,0])
            pow_img_lr = array2sitk(pow_lr, [1,1], [0,0])
            diffSpeImage = array2sitk(diff_spe, [1,1], [0,0])
            hr_spe_img = array2sitk(hr_spe, [1,1], [0,0])
            sr_spe_img = array2sitk(sr_spe, [1,1], [0,0])
            sitk.WriteImage(pow_img_sr, '{}/{}-{}_sr_pow.mhd'.format(result_path2, path_i, img_i))
            sitk.WriteImage(pow_img_hr, '{}/0-{}_hr_pow.mhd'.format(result_path2, img_i))
            sitk.WriteImage(pow_img_lr, '{}/0-{}_lr_pow.mhd'.format(result_path2, img_i))
            sitk.WriteImage(diffSpeImage, '{}/{}-{}-diff_power_spe.mhd'.format(result_path2, path_i, img_i))
            sitk.WriteImage(hr_spe_img, '{}/0-{}-hr_spe.mhd'.format(result_path2, img_i))
            sitk.WriteImage(sr_spe_img, '{}/{}-{}-sr_spe.mhd'.format(result_path2, path_i, img_i))
        plt.legend()
        plt.grid()
        plt.yscale('log')
        plt.xlabel("Frequency")
        plt.ylabel("Normalized Power Spectrum")
        plt.title("Frequency Spectrum")
        plt.savefig("{}/frequency_spectrum_without_log_sub_org{}.png".format(result_path, img_i))
        plt.clf()