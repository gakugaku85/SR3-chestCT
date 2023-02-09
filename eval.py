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

    result_path = './experiments/eval2/'
    result_path1 = result_path + 'result'
    result_path2 = result_path + 'spect'
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(result_path1, exist_ok=True)
    os.makedirs(result_path2, exist_ok=True)
    ipx = 0
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    comp_hr, comp_sr, comp_lr = [], [], []
    for test_path in test_paths:
        ipx+=1
        real_names = natsorted(glob(osp.join('{}/*_hr.mhd'.format(test_path))))
        fake_names = natsorted(glob(osp.join('{}/*_sr.mhd'.format(test_path))))
        lr_names = natsorted(glob(osp.join('{}/*_inf.mhd'.format(test_path))))

        hr_imgs = []
        lr_imgs = []
        sr_imgs = []
        hr_patchs = []
        lr_patchs = []
        
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

            Metrics.save_mhd(hr_sobel, '{}/0_{}_sobelHR.mhd'.format(result_path1, idx))
            Metrics.save_mhd(lr_sobel, '{}/0_{}_sobelLR.mhd'.format(result_path1, idx))
            Metrics.save_mhd(ac_map_img, '{}/0_{}_mapHRLR.mhd'.format(result_path1, idx))
            Metrics.save_mhd(sr_sobel, '{}/{}_{}_sobelSR.mhd'.format(result_path1, ipx, idx))
            Metrics.save_mhd(sobel_map_img, '{}/{}_{}_sobelMAP.mhd'.format(result_path1, ipx, idx))

            Metrics.save_mhd(map_img, '{}/{}_{}_map.mhd'.format(result_path1, ipx, idx))
            Metrics.save_mhd(patch_map_img, '{}/{}_{}_patch_map.mhd'.format(result_path1, ipx, idx))

            Metrics.save_mhd(lr_patch, '{}/0_{}_lr.mhd'.format(result_path1, idx))
            Metrics.save_mhd(hr_patch, '{}/0_{}_hr.mhd'.format(result_path1, idx))
            Metrics.save_mhd(sr_patch, '{}/{}_{}sr.mhd'.format(result_path1, ipx, idx))
            # idx+=1
        comp_hr.append(hr_imgs)
        comp_sr.append(sr_imgs)
        comp_lr.append(lr_imgs)


    compare_len = len(comp_hr)
    data_length = len(hr_imgs)
    for j in range(compare_len):
        results = []
        m_i = 0
        hr_imgs = comp_hr[j]
        lr_imgs = comp_lr[j]
        sr_imgs = comp_sr[j]
        for mask_imgs in mask_list:
            avg_psnr = 0.0
            avg_ssim = 0.0
            avg_zncc = 0.0
            avg_dpow = 0.0
            for i in range(data_length):
                psnr = Metrics.calculate_psnr_mask(hr_imgs[i], sr_imgs[i], mask_imgs[i])
                ssim = Metrics.calculate_ssim_mask(hr_imgs[i], sr_imgs[i], mask_imgs[i])
                zncc = Metrics.calc_zncc(hr_imgs[i], sr_imgs[i])
                diff_spe_const, diff_spe, sr_spe, hr_spe = Metrics.calc_fft_domain(hr_imgs[i], sr_imgs[i])
                result = {"PSNR": psnr, "SSIM": ssim, "ZNCC": zncc, "MASK":mask_list_path[m_i], 'MAE-Power':diff_spe_const}

                h, w = hr_imgs[i].shape
                size = 256

                # 画像の中心を計算する
                center_y = h // 2
                center_x = w // 2

                hr_fft = hr_imgs[i][center_y-size:center_y+size, center_x-size:center_x+size].astype(np.float32)
                sr_fft = sr_imgs[i][center_y-size:center_y+size, center_x-size:center_x+size].astype(np.float32)
                hr_fft = (hr_fft - hr_fft.min()) / (hr_fft.max() - hr_fft.min())
                sr_fft = (sr_fft - sr_fft.min()) / (sr_fft.max() - sr_fft.min())
                hr_fft = hr_fft * 2 - 1
                sr_fft = sr_fft * 2 - 1

                f_hr = np.fft.fftn(hr_fft)
                f_sr = np.fft.fftn(sr_fft)
                fshift_hr = np.fft.fftshift(f_hr)
                fshift_sr = np.fft.fftshift(f_sr)

                magnitude_spectrum_hr = 20 * np.log(np.abs(fshift_hr))
                magnitude_spectrum_sr = 20 * np.log(np.abs(fshift_sr))

                data = np.mean(magnitude_spectrum_hr, axis=1)[size:]
                average_hr = np.convolve(data, np.ones(10) / 10, mode='valid')
                data = np.mean(magnitude_spectrum_sr, axis=1)[size:]
                average_sr = np.convolve(data, np.ones(10) / 10, mode='valid')

                ix = 5
                if i == ix and m_i == 0:
                    if i == ix and m_i == 0 and j == 0:
                        plt.plot(average_hr, label='hr_spect', linewidth=1)
                    plt.plot(average_sr, label='sr_spect{}'.format(j+1), linewidth=1)
                    plt.legend()
                    plt.xlabel("Frequency")
                    plt.ylabel("Normalized Power Spectrum")
                    plt.title("Frequency Spectrum")
                    plt.savefig("{}/frequency_spectrum{}.png".format(result_path, i))

                results.append(result)
                print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}, ZNCC:{:.4f}'.format(i, psnr, ssim, zncc))

                def array2sitk(arr, spacing=[], origin=[]):
                    if not len(spacing) == arr.ndim and len(origin) == arr.ndim:
                        print("Dimension Error")
                        quit()
                    sitkImg = sitk.GetImageFromArray(arr)
                    sitkImg.SetSpacing(spacing)
                    sitkImg.SetOrigin(origin)
                    return sitkImg

                power_spe = array2sitk(magnitude_spectrum_hr, [1,1], [0,0])
                pow_image = array2sitk(magnitude_spectrum_sr, [1,1], [0,0])
                diffSpeImage = array2sitk(diff_spe, [1,1], [0,0])
                hr_spe_img = array2sitk(hr_spe, [1,1], [0,0])
                sr_spe_img = array2sitk(sr_spe, [1,1], [0,0])
                sitk.WriteImage(pow_image, '{}/{}-{}_sr_pow.mhd'.format(result_path2, j, i))
                sitk.WriteImage(power_spe, '{}/0-{}_pow.mhd'.format(result_path2, i))
                sitk.WriteImage(diffSpeImage, '{}/{}-{}-diff_power_spe.mhd'.format(result_path2, j, i))
                sitk.WriteImage(hr_spe_img, '{}/0-{}-hr_spe.mhd'.format(result_path2, i))
                sitk.WriteImage(sr_spe_img, '{}/{}-{}-sr_spe.mhd'.format(result_path2, j, i))
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
            m_i += 1
        df = pd.DataFrame(data=results)
        df.to_csv(result_path+"result{}.csv".format(j), index=False)

