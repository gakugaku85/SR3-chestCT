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
from torchvision.transforms import functional as trans_fn

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

    mask_root_path = "../dataset/mask_1792"
    mask_list_path = ['_png_2/*.png', '_sobel_png_1/*.png', '_sobel_png_2/*.png', '_sobel_png_3/*.png', '_sobel_png_1_outside/*.png', '_sobel_png_E/*.png']
    
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

    result_path = './experiments/eval/'
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
            H, W = 120, 120
            hr_patch = img_HR[patch_h:patch_h+H, patch_w: patch_w+W]
            sr_patch = img_SR[patch_h:patch_h+H, patch_w: patch_w+W]
            lr_patch = img_LR[patch_h:patch_h+H, patch_w: patch_w+W]

            mix_sr_img = checkerboard(hr_patch, sr_patch, 18)
            mix_lr_img = checkerboard(hr_patch, lr_patch, 18)

            pil_image = Image.fromarray(hr_patch)
            LR_patch = trans_fn.resize(pil_image, (H//4, W//4), Image.BICUBIC)

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

            Metrics.save_img(hr_sobel, '{}/0_{}_sobelHR.png'.format(result_path1, idx))
            Metrics.save_img(lr_sobel, '{}/0_{}_sobelLR.png'.format(result_path1, idx))
            Metrics.save_img(ac_map_img, '{}/0_{}_mapHRLR.png'.format(result_path1, idx))
            Metrics.save_img(sr_sobel, '{}/{}_{}_sobelSR.png'.format(result_path1, ipx, idx))
            Metrics.save_img(sobel_map_img, '{}/{}_{}_sobelMAP.png'.format(result_path1, ipx, idx))

            Metrics.save_img(map_img, '{}/{}_{}_map.png'.format(result_path1, ipx, idx))
            Metrics.save_img(patch_map_img, '{}/{}_{}_patch_map.png'.format(result_path1, ipx, idx))

            Metrics.save_img(lr_patch, '{}/0_{}_lr.png'.format(result_path1, idx))
            Metrics.save_img(hr_patch, '{}/0_{}_hr.png'.format(result_path1, idx))
            Metrics.save_img(np.array(LR_patch), '{}/0_{}_LR_2.png'.format(result_path1, idx))
            Metrics.save_img(sr_patch, '{}/{}_{}sr.png'.format(result_path1, ipx, idx))
            Metrics.save_img(mix_sr_img, '{}/{}_{}_mixSR.png'.format(result_path1, ipx, idx))
            Metrics.save_img(mix_lr_img, '{}/0_{}_mixLR.png'.format(result_path1, idx))
            idx+=1
        ipx+=1
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
                result = {"PSNR": psnr, "SSIM": ssim, "ZNCC": zncc, "MASK":mask_list_path[m_i], 'D-Power':diff_spe_const}

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
            m_i += 1
        df = pd.DataFrame(data=results)
        df.to_csv(result_path+"result{}.csv".format(j), index=False)

