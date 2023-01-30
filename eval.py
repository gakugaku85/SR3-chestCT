import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
import os
import SimpleITK as sitk
from natsort import natsorted

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments/sr_microCT_patch_230112_140212/results/3134')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.mhd'.format(args.path)))
    fake_names = list(glob.glob('{}/*_sr.mhd'.format(args.path)))
    lr_names = list(glob.glob('{}/*_inf.mhd'.format(args.path)))

    real_names.sort()
    fake_names.sort()
    mask_root_path = "../dataset/mask_1794_sobel_png_2"
    mask_imgs = []
    for mask_filename in natsorted(os.listdir(os.path.abspath(mask_root_path))):
        mask_path = os.path.join(mask_root_path, mask_filename)
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask, dtype=np.uint8)
        mask_imgs.append(mask)
        print(mask_path)

    print(real_names)

    result_path = 'experiments/validation/sr_microCT_patch_val_230119_171947/results'

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_zncc = 0.0
    avg_dpow = 0.0
    idx = 0
    for rname, fname, lname in zip(real_names, fake_names, lr_names):
        idx += 1
        ridx = rname.rsplit("_hr")[0]
        fidx = rname.rsplit("_sr")[0]

        img_HR = sitk.ReadImage(rname)
        img_SR = sitk.ReadImage(fname)
        img_LR = sitk.ReadImage(lname)

        # sobel_HR = sitk.SobelEdgeDetection(img_HR)
        # sobel_SR = sitk.SobelEdgeDetection(img_SR)
        # sitk.WriteImage(sobel_HR, '{}/{}_sobelHR.mhd'.format(result_path, idx))
        # sitk.WriteImage(sobel_SR, '{}/{}_sobelSR.mhd'.format(result_path, idx))

        hr_img = sitk.GetArrayFromImage(img_HR)
        sr_img = sitk.GetArrayFromImage(img_SR)
        lr_img = sitk.GetArrayFromImage(img_LR)

        mask = mask_imgs[idx-1]
        psnr = Metrics.calculate_psnr_mask(sr_img, hr_img, mask)
        ssim = Metrics.calculate_ssim_mask(sr_img, hr_img, mask)
        zncc = Metrics.zncc(sr_img, hr_img)
        dpow = Metrics.d_power(sr_img, hr_img)
        avg_psnr += psnr
        avg_ssim += ssim
        avg_zncc += zncc
        avg_dpow += dpow
        print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}, ZNCC:{:.4f}, D-power:{:.4f}'.format(idx, psnr, ssim, zncc, dpow))
        # map_img = (sr_img - hr_img)**2
        # Metrics.save_mhd(map_img, '{}/{}_2map.mhd'.format(result_path, idx))

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_zncc = avg_zncc / idx
    avg_dpow = avg_dpow / idx

    # log
    print('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    print('# Validation # SSIM: {:.4e}'.format(avg_ssim))
    print('# Validation # ZNCC: {:.4e}'.format(avg_zncc))
    print('# Validation # D-power: {:.4e}'.format(avg_dpow))

