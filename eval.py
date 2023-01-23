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
                        default='experiments/sr_microCT_patch_val_230119_171947/results')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.mhd'.format(args.path)))
    fake_names = list(glob.glob('{}/*_sr.mhd'.format(args.path)))
    lr_names = list(glob.glob('{}/*_inf.mhd'.format(args.path)))

    real_names.sort()
    fake_names.sort()

    mask_imgs = []
    for mask_filename in natsorted(os.listdir(os.path.abspath("../dataset/mask_1792_png_2"))):
        mask_path = os.path.join("../dataset/mask_1792_png_2", mask_filename)
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask, dtype=np.uint8)
        mask_imgs.append(mask)
        print(mask_path)

    print(real_names)

    result_path = 'experiments/sr_microCT_patch_val_230119_171947/results'

    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    for rname, fname, lname in zip(real_names, fake_names, lr_names):
        idx += 1
        ridx = rname.rsplit("_hr")[0]
        fidx = rname.rsplit("_sr")[0]
        # assert ridx == fidx, 'Image ridx:{ridx}!=fidx:{fidx}'.format(ridx, fidx)

        img_HR = sitk.ReadImage(rname)
        img_SR = sitk.ReadImage(fname)
        img_LR = sitk.ReadImage(lname)

        sobel_HR = sitk.SobelEdgeDetection(img_HR)
        sobel_SR = sitk.SobelEdgeDetection(img_SR)
        sitk.WriteImage(sobel_HR, '{}/{}_sobelHR.mhd'.format(result_path, idx))
        sitk.WriteImage(sobel_SR, '{}/{}_sobelSR.mhd'.format(result_path, idx))

        hr_img = sitk.GetArrayFromImage(img_HR)
        sr_img = sitk.GetArrayFromImage(img_SR)
        lr_img = sitk.GetArrayFromImage(img_LR)

        # hr_img = np.array(Image.open(rname))
        # sr_img = np.array(Image.open(fname))
        mask = mask_imgs[idx-1]
        psnr = Metrics.calculate_psnr_mask(sr_img, hr_img, mask)
        ssim = Metrics.calculate_ssim_mask(sr_img, hr_img, mask)
        avg_psnr += psnr
        avg_ssim += ssim
        print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}'.format(idx, psnr, ssim))
        # if idx % 20 == 0:
        map_img = (sr_img - hr_img)**2
        Metrics.save_mhd(map_img, '{}/{}_2map.mhd'.format(result_path, idx))

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx

    # log
    print('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    print('# Validation # SSIM: {:.4e}'.format(avg_ssim))
