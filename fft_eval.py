import argparse
import logging
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import core.metrics as Metrics


def compute_mean_values_by_distance(image, num_pixels):
    """距離ごとの平均値を計算

    Args:
        image (numpy.ndarray): 画像データ（フーリエ変換後のパワースペクトル）
        num_pixels (int): 計算する距離の数

    Returns:
        tuple: 距離ごとの平均値とNyquist周波数での平均値
    """
    height, width = image.shape
    center_x, center_y = width / 2 + 0.5, height / 2 + 0.5
    values, Nyquist_mean = [], []
    sum_value_ny = 0
    distances = np.zeros((height, width), dtype=int)

    for y in range(height):
        for x in range(width):
            distances[y, x] = max(abs(x - center_x), abs(y - center_y))

    for i in range(num_pixels):
        sum_values, pixels = 0, 0
        for y in range(height):
            for x in range(width):
                if distances[y, x] == i:
                    sum_values += image[y, x]
                    pixels += 1
        mean_value = sum_values / pixels if pixels != 0 else 0
        values.append(mean_value)
        sum_value_ny += mean_value
        if (i + 1) % (num_pixels // 4) == 0 and i != 0:
            Nyquist_mean.append(sum_value_ny / (num_pixels // 4))
            sum_value_ny = 0
    return values, Nyquist_mean

def array2sitk(arr, spacing=[], origin=[]):
                if not len(spacing) == arr.ndim and len(origin) == arr.ndim:
                    print("Dimension Error")
                    quit()
                sitkImg = sitk.GetImageFromArray(arr)
                sitkImg.SetSpacing(spacing)
                sitkImg.SetOrigin(origin)
                return sitkImg

def save_power_spectrum_as_png(power_spectrum, output_path):
    """パワースペクトルをPNG画像として保存

    Args:
        power_spectrum (numpy.ndarray): パワースペクトル
        output_path (str): 出力パス
    """
    plt.imshow(10*np.log1p(power_spectrum), cmap='gray')
    plt.axis('off')  # 軸を表示しない
    plt.tight_layout()
    try:
        plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)
    except Exception as e:
        logging.error(f"パワースペクトルの保存に失敗しました: {output_path}, エラー: {e}")

def save_org_img_as_png(img, output_path):
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # 軸を表示しない
    plt.tight_layout()
    try:
        plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)
    except Exception as e:
        logging.error(f"パワースペクトルの保存に失敗しました: {output_path}, エラー: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments/fft_eval/')
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
                'experiments/val_wd_10_best/results']

    sr_title = ['ours(original loss)', 'ours(wd loss)']

    result_path = './experiments/eval_fft/'
    result_path1 = result_path + 'result'
    result_path2 = result_path + 'spect'
    result_path_png = result_path + 'png'
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(result_path1, exist_ok=True)
    os.makedirs(result_path2, exist_ok=True)
    os.makedirs(result_path_png, exist_ok=True)

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
        ny_sub = []
        for path_i in range(len(comp_hr_names)):
            img_HR = sitk.GetArrayFromImage(sitk.ReadImage(comp_hr_names[path_i][img_i]))
            img_SR = sitk.GetArrayFromImage(sitk.ReadImage(comp_sr_names[path_i][img_i]))
            img_LR = sitk.GetArrayFromImage(sitk.ReadImage(comp_lr_names[path_i][img_i]))
            h, w = img_HR.shape
            center_y = h // 2
            center_x = w // 2
            size = 128
            hr_org = img_HR[center_y-size:center_y+size, center_x-size:center_x+size].astype(np.float32)
            sr_org = img_SR[center_y-size:center_y+size, center_x-size:center_x+size].astype(np.float32)
            lr_org = img_LR[center_y-size:center_y+size, center_x-size:center_x+size].astype(np.float32)

            # save_org_img_as_png(hr_org, '{}/{}-hr_org.png'.format(result_path_png, img_i))
            # save_org_img_as_png(sr_org, '{}/{}-{}_sr_org.png'.format(result_path_png, img_i, sr_title[path_i].split("(")[1].split(" ")[0]))
            # save_org_img_as_png(lr_org, '{}/{}-lr_org.png'.format(result_path_png, img_i))

            diff_spe_const, diff_spe, hr_spe, sr_spe = Metrics.calc_fft_domain(hr_org, sr_org)

            Metrics.save_mhd(img_HR, '{}/0_{}hr_org.mhd'.format(result_path2, img_i))
            Metrics.save_mhd(img_LR, '{}/0_{}lr_org.mhd'.format(result_path2, img_i))
            Metrics.save_mhd(img_SR, '{}/{}_{}sr_org.mhd'.format(result_path2, path_i, img_i))

            # hr_org = (hr_org - hr_org.min()) / (hr_org.max() - hr_org.min())
            # sr_org = (sr_org - sr_org.min()) / (sr_org.max() - sr_org.min())
            # lr_org = (lr_org - lr_org.min()) / (lr_org.max() - lr_org.min())
            # hr_org = hr_org * 2 - 1
            # sr_org = sr_org * 2 - 1
            # lr_org = lr_org * 2 - 1

            pow_hr = np.abs(np.fft.fftshift(np.fft.fftn(hr_org)))**2
            pow_sr = np.abs(np.fft.fftshift(np.fft.fftn(sr_org)))**2
            pow_lr = np.abs(np.fft.fftshift(np.fft.fftn(lr_org)))**2

            # pow_hr = 10 * np.log(pow_hr)
            # pow_sr = 10 * np.log(pow_sr)
            # pow_lr = 10 * np.log(pow_lr)

            if path_i == 0:
                freq_hr, ny_hr = Metrics.mean_values_by_distance(pow_hr, size)
                freq_lr, ny_lr = Metrics.mean_values_by_distance(pow_lr, size) #lrを表示する場合
            freq_sr, ny_sr = Metrics.mean_values_by_distance(pow_sr, size)

            ny_sub.append([(b-a)/b for a, b in zip(ny_hr, ny_sr)]) #ナイキスト周波数までの差分を計算
            if path_i == 0:
                # plt.plot(freq_hr, label='GT', linewidth=1)
                # plt.plot(freq_lr, label='lr_spect', linewidth=1) #lrを表示する場合
                plt.plot( [(lr - hr) / hr for lr, hr in zip(freq_lr, freq_hr)], label='lr_spect', linewidth=1) #lrを表示する場合

            # plt.plot(freq_sr, label=sr_title[path_i], linewidth=1)
            plt.plot( [(sr - hr) / hr for sr, hr in zip(freq_sr, freq_hr)], label=sr_title[path_i], linewidth=1) #lrを表示する場合


            # print([np.abs(a-b) for a, b in zip(ny_hr, ny_sr)])

            # if path_i == 0: #差分を表示する場合
            #     freq_hr, ny_hr= Metrics.mean_values_by_distance(pow_hr,size)
            #     # freq_lr = Metrics.mean_values_by_distance(pow_lr, size)
            # freq_sr, ny_sr = Metrics.mean_values_by_distance(pow_sr, size)
            # freq_sub_srhr = [np.abs(a - b) for a, b in zip(freq_hr, freq_sr)]
            # plt.plot(freq_sub_srhr, label='sr_sub_spect{}'.format(path_i+1), linewidth=1)

            pow_img_hr = array2sitk(pow_hr, [1,1], [0,0])
            pow_img_sr = array2sitk(pow_sr, [1,1], [0,0])
            pow_img_lr = array2sitk(pow_lr, [1,1], [0,0])
            # diffSpeImage = array2sitk(diff_spe, [1,1], [0,0])
            # hr_spe_img = array2sitk(hr_spe, [1,1], [0,0])
            # sr_spe_img = array2sitk(sr_spe, [1,1], [0,0])
            sitk.WriteImage(pow_img_sr, '{}/{}-{}_sr_pow.mhd'.format(result_path2, path_i, img_i))
            sitk.WriteImage(pow_img_hr, '{}/0-{}_hr_pow.mhd'.format(result_path2, img_i))
            sitk.WriteImage(pow_img_lr, '{}/0-{}_lr_pow.mhd'.format(result_path2, img_i))

            # save_power_spectrum_as_png(pow_hr, '{}/{}-hr_pow.png'.format(result_path_png, img_i))
            # save_power_spectrum_as_png(pow_sr, '{}/{}-{}_sr_pow.png'.format(result_path_png, img_i, sr_title[path_i].split("(")[1].split(" ")[0]))
            # save_power_spectrum_as_png(pow_lr, '{}/{}-lr_pow.png'.format(result_path_png, img_i))

            # sitk.WriteImage(diffSpeImage, '{}/{}-{}-diff_power_spe.mhd'.format(result_path2, path_i, img_i))
            # sitk.WriteImage(hr_spe_img, '{}/0-{}-hr_spe.mhd'.format(result_path2, img_i))
            # sitk.WriteImage(sr_spe_img, '{}/{}-{}-sr_spe.mhd'.format(result_path2, path_i, img_i))
        plt.legend()
        plt.grid()
        # plt.yscale('log')
        plt.xlabel("harmonic num.")
        plt.ylabel("Power Spectrum")
        plt.ylabel("Relative error with power spectrum of HRimage")
        plt.title("Frequency Spectrum")
        plt.savefig("{}/frequency_spectrum{}sum.png".format(result_path1, img_i))
        plt.clf()

        df = pd.DataFrame(data=ny_sub)
        df.to_csv(result_path1+"/nyquist{}.csv".format(img_i), index=False)
        print("finish{}".format(img_i))