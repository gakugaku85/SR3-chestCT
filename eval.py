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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments/sr_sam_dan/my_results')
    args = parser.parse_args()

    mask_root_path = "../dataset/mask_1792"
    mask_list_path = ['_png_2/*.png', '_sobel_png_1/*.png', '_sobel_png_E/*.png', '_sobel_png_1_outside/*.png']
    
    mask_files = [natsorted(glob(osp.join(mask_root_path+path))) for path in mask_list_path]

    mask_list = []
    for mask_file in mask_files:
        mask_imgs = []
        for mask_path in mask_file:
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask, dtype=np.uint8)
            mask_imgs.append(mask)
        mask_list.append(mask_imgs)

    test_paths =['experiments/sr_microCT_patch_val_230202_073558/results',
                'experiments/validation/sr_microCT_patch_val_230118_071603/results',
                'experiments/sr_sam_dan/my_results',
                # 'experiments/sr_microCT_patch_val_230202_063852/results',
                'experiments/sr_microCT_patch_val_230202_073935/results']
                # 'experiments/sr_microCT_patch_val_230202_074137/results',
                # 'experiments/sr_microCT_patch_val_230202_075135/results',
    result_path = './experiments/all4/'
    os.makedirs(result_path, exist_ok=True)
    ipx = 0
    comp_imgs = []
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    for test_path in test_paths:
        ipx+=1
        real_names = list(glob('{}/*_hr.mhd'.format(test_path)))
        fake_names = list(glob('{}/*_sr.mhd'.format(test_path)))
        lr_names = list(glob('{}/*_inf.mhd'.format(test_path)))
        real_names.sort()
        fake_names.sort()
        lr_names.sort()

        hr_imgs = []
        lr_imgs = []
        sr_imgs = []
        hr_patchs = []
        lr_patchs = []
        compare_img = []
        idx = 0
        for rname, fname, lname in zip(real_names, fake_names, lr_names): 
            img_HR = sitk.GetArrayFromImage(sitk.ReadImage(rname))
            img_SR = sitk.GetArrayFromImage(sitk.ReadImage(fname))
            img_LR = sitk.GetArrayFromImage(sitk.ReadImage(lname))
            hr_imgs.append(img_HR)
            sr_imgs.append(img_SR)
            lr_imgs.append(img_LR)
            patch_h, patch_w = 1000, 520
            H, W = 180, 120
            hr_patch = img_HR[patch_h:patch_h+H, patch_w: patch_w+W]
            sr_patch = img_SR[patch_h:patch_h+H, patch_w: patch_w+W]
            lr_patch = img_LR[patch_h:patch_h+H, patch_w: patch_w+W]

            sobelx = ndimage.convolve(hr_patch, sobel_x)
            sobely = ndimage.convolve(hr_patch, sobel_y)
            hr_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

            sobelx = ndimage.convolve(lr_patch, sobel_x)
            sobely = ndimage.convolve(lr_patch, sobel_y)
            lr_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

            sobelx = ndimage.convolve(sr_patch, sobel_x)
            sobely = ndimage.convolve(sr_patch, sobel_y)
            sr_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

            sobel_map_img = (hr_sobel - sr_sobel)**2
            ac_map_img = (hr_sobel - lr_sobel)**2
            patch_map_img = (hr_patch - sr_patch)**2
            map_img = (img_HR - img_SR)**2

            Metrics.save_mhd(hr_sobel, '{}/0_{}_sobelHR.mhd'.format(result_path, idx))
            Metrics.save_mhd(lr_sobel, '{}/0_{}_sobelLR.mhd'.format(result_path, idx))
            Metrics.save_mhd(ac_map_img, '{}/0_{}_mapHRLR.mhd'.format(result_path, idx))
            Metrics.save_mhd(sr_sobel, '{}/{}_{}_sobelSR.mhd'.format(result_path, ipx, idx))
            Metrics.save_mhd(sobel_map_img, '{}/{}_{}_sobelMAP.mhd'.format(result_path, ipx, idx))

            Metrics.save_mhd(map_img, '{}/{}_{}_map.mhd'.format(result_path, ipx, idx))
            Metrics.save_mhd(patch_map_img, '{}/{}_{}_patch_map.mhd'.format(result_path, ipx, idx))

            Metrics.save_mhd(lr_patch, '{}/0_{}_lr.mhd'.format(result_path, idx))
            Metrics.save_mhd(hr_patch, '{}/0_{}_hr.mhd'.format(result_path, idx))
            Metrics.save_mhd(sr_patch, '{}/{}_{}sr.mhd'.format(result_path, ipx, idx))
            idx+=1



    results = []
    data_length = len(hr_imgs)
    m_i = 0

    for mask_imgs in mask_list:
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_zncc = 0.0
        for i in range(data_length):
            psnr = Metrics.calculate_psnr_mask(lr_imgs[i], hr_imgs[i], mask_imgs[i])
            ssim = Metrics.calculate_ssim_mask(lr_imgs[i], hr_imgs[i], mask_imgs[i])
            zncc = Metrics.zncc(lr_imgs[i], hr_imgs[i])
            result = {"PSNR": psnr, "SSIM": ssim, "ZNCC": zncc, "MASK":mask_list_path[m_i]}

            results.append(result)
            print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}, ZNCC:{:.4f}'.format(i, psnr, ssim, zncc))
            avg_psnr += psnr
            avg_ssim += ssim
            avg_zncc += zncc

        avg_psnr = avg_psnr / data_length
        avg_ssim = avg_ssim / data_length
        avg_zncc = avg_zncc / data_length

        # log
        result = {"avePSNR": avg_psnr, "aveSSIM": avg_ssim, "aveZNCC": avg_zncc}
        results.append(result)
        print('Validation # PSNR: {:.4e}'.format(avg_psnr))
        print('Validation # SSIM: {:.4e}'.format(avg_ssim))
        print('Validation # ZNCC: {:.4e}'.format(avg_zncc))
        m_i += 1
    df = pd.DataFrame(data=results)
    df.to_csv(args.path+"result_LR.csv", index=False)

