import argparse
import os
import os.path as osp
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from natsort import natsorted
from PIL import Image
from scipy import ndimage
from torchvision.transforms import functional as trans_fn

import core.metrics as Metrics


def checkerboard(img_hr, img_sr, stride):
    i, j = 0, 0
    H, W = img_hr.shape
    mix_img = np.zeros((H+stride, W+stride), dtype=np.float64)
    pad_hr = np.zeros((H+stride, W+stride), dtype=np.float64)
    pad_sr = np.zeros((H+stride, W+stride), dtype=np.float64)
    pad_hr[0:H, 0:W] = img_hr
    pad_sr[0:H, 0:W] = img_sr
    for h in range(0, H, stride):
        i = j%2
        for w in range(0, W, stride):
            if i % 2 == 0 :
                mix_img[h:h+stride, w:w+stride] = pad_hr[h:h+stride, w:w+stride]
            else:
                mix_img[h:h+stride, w:w+stride] = pad_sr[h:h+stride, w:w+stride]
            i+=1
        j += 1
    return mix_img[0:H, 0:W]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments/sr_sam_dan/my_results')
    args = parser.parse_args()

    mask_root_path = "/take/dataset/microCT_slices_1792_2/mask/"

    mask_file_path = natsorted(glob(osp.join(mask_root_path, '*.png')))
    print(mask_file_path)

    mask_imgs = []
    for mask_path in mask_file_path:
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask, dtype=np.uint8)
        mask_imgs.append(mask)

    test_paths =['experiments/val_ori_best/results',
                'experiments/val_wd_10_best/results',
                'experiments/val_wd_100_best/results']

    comp_hr, comp_sr, comp_lr = [], [], []
    for ipx, test_path in enumerate(test_paths):
        result_path = test_path
        result_path2 = result_path + '/spect'
        os.makedirs(result_path, exist_ok=True)
        os.makedirs(result_path2, exist_ok=True)

        real_names = natsorted(glob(osp.join('{}/*_hr.mhd'.format(test_path))))
        fake_names = natsorted(glob(osp.join('{}/*_sr.mhd'.format(test_path))))
        lr_names = natsorted(glob(osp.join('{}/*_inf.mhd'.format(test_path))))

        print(real_names)

        hr_imgs, lr_imgs, sr_imgs = [], [], []

        for rname, fname, lname in zip(real_names, fake_names, lr_names):
            img_HR = sitk.GetArrayFromImage(sitk.ReadImage(rname))
            img_SR = sitk.GetArrayFromImage(sitk.ReadImage(fname))
            img_LR = sitk.GetArrayFromImage(sitk.ReadImage(lname))
            hr_imgs.append(img_HR)
            sr_imgs.append(img_SR)
            lr_imgs.append(img_LR)

        comp_hr.append(hr_imgs)
        comp_sr.append(sr_imgs)
        comp_lr.append(lr_imgs)


    compare_len = len(comp_hr)
    data_length = len(hr_imgs)
    for j in range(compare_len):
        results = []
        hr_imgs = comp_hr[j]
        lr_imgs = comp_lr[j]
        sr_imgs = comp_sr[j]

        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_zncc = 0.0
        avg_dpow = 0.0
        for i in range(data_length):
            psnr = Metrics.calculate_psnr_mask(hr_imgs[i], sr_imgs[i], mask_imgs[i])
            ssim = Metrics.calculate_ssim_mask(hr_imgs[i], sr_imgs[i], mask_imgs[i])
            zncc = Metrics.calc_zncc(hr_imgs[i], sr_imgs[i])

            h, w = hr_imgs[i].shape
            size = 128
            patch_size = 42

            # 画像の中心を計算する
            center_y = h // 2
            center_x = w // 2

            hr_org = hr_imgs[i][center_y-size:center_y+size, center_x-size:center_x+size].astype(np.float32)
            sr_org = sr_imgs[i][center_y-size:center_y+size, center_x-size:center_x+size].astype(np.float32)
            lr_org = lr_imgs[i][center_y-size:center_y+size, center_x-size:center_x+size].astype(np.float32)
            diff_spe_const, diff_spe, sr_spe, hr_spe = Metrics.calc_fft_domain_mask(hr_imgs[i], sr_imgs[i], mask_imgs[i])
            result = {"PSNR": psnr, "SSIM": ssim, "ZNCC": zncc, "MASK":0, 'D-Power':diff_spe_const}

            hr_org_p = hr_imgs[i][center_y-patch_size:center_y+patch_size, center_x-patch_size:center_x+patch_size].astype(np.float32)
            sr_org_p = sr_imgs[i][center_y-patch_size:center_y+patch_size, center_x-patch_size:center_x+patch_size].astype(np.float32)
            lr_org_p = lr_imgs[i][center_y-patch_size:center_y+patch_size, center_x-patch_size:center_x+patch_size].astype(np.float32)

            Metrics.save_img(hr_org, '{}/0_{}hr_org.png'.format(result_path2, i))
            Metrics.save_img(lr_org, '{}/0_{}lr_org.png'.format(result_path2, i))
            Metrics.save_img(sr_org, '{}/{}_{}sr_org.png'.format(result_path2, j, i))

            Metrics.save_img(hr_org_p, '{}/0_{}hr_org_p.png'.format(result_path2, i))
            Metrics.save_img(lr_org_p, '{}/0_{}lr_org_p.png'.format(result_path2, i))
            Metrics.save_img(sr_org_p, '{}/{}_{}sr_org_p.png'.format(result_path2, j, i))

            results.append(result)
            print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}, D-pow:{:.4f}'.format(i, psnr, ssim, diff_spe_const))

            avg_psnr += psnr
            avg_ssim += ssim
            avg_zncc += zncc
            avg_dpow += diff_spe_const

        avg_psnr = avg_psnr / data_length
        avg_ssim = avg_ssim / data_length
        avg_zncc = avg_zncc / data_length
        avg_dpow = avg_dpow / data_length

        # log
        result = {"avePSNR": avg_psnr, "aveSSIM": avg_ssim, "aveZNCC": avg_zncc, "aveDpow": avg_dpow}
        results.append(result)
        print('Validation # PSNR: {:.4e}'.format(avg_psnr))
        print('Validation # SSIM: {:.4e}'.format(avg_ssim))
        print('Validation # ZNCC: {:.4e}'.format(avg_zncc))
        df = pd.DataFrame(data=results)
        df.to_csv(result_path+"result{}.csv".format(j), index=False)

