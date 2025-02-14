import argparse
import json
import os
import os.path as osp
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
from icecream import ic
from natsort import natsorted
from scipy.stats import wilcoxon
from skimage import io
from skimage.filters import frangi
from skimage.metrics import mean_squared_error


def load_mhd(file_path):
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)  # SimpleITKイメージをNumPy配列に変換
    return array, image

def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    dir_list = [
    "experiments/val_ori_best/results",
    "experiments/val_wd_10_best/results"
    ]
    json_path = "/take/dataset/microCT_slices_1792_2/patch_metadata_128.json"

    thresholds = [0, 0.01, 0.05, 0.1, 0.2, 0.3]
    patch_data_list = load_json(json_path)

    mse_data_list = []

    for dir in dir_list:
        test_path = dir

        ic(test_path)
        hr_paths = natsorted(glob(osp.join(test_path, '*_hr.mhd')))
        sr_paths = natsorted(glob(osp.join(test_path, '*_sr.mhd')))
        lr_paths = natsorted(glob(osp.join(test_path, '*_inf.mhd')))

        hr_imgs = [load_mhd(hr_path)[0] for hr_path in hr_paths]
        sr_imgs = [load_mhd(sr_path)[0] for sr_path in sr_paths]
        lr_imgs = [load_mhd(lr_path)[0] for lr_path in lr_paths]

        patch_mse_list = []

        for i, patch_data in enumerate(patch_data_list):
            file_name = patch_data["file_path"].split("/")[-1]
            img_idx = int(file_name.split(".")[0])/100

            hr_img = hr_imgs[int(img_idx)-1]
            sr_img = sr_imgs[int(img_idx)-1]
            lr_img = lr_imgs[int(img_idx)-1]

            hr_patch = hr_img[patch_data["top_left_y"]:patch_data["top_left_y"]+patch_data["patch_size"][0], patch_data["top_left_x"]:patch_data["top_left_x"]+patch_data["patch_size"][1]]
            sr_patch = sr_img[patch_data["top_left_y"]:patch_data["top_left_y"]+patch_data["patch_size"][0], patch_data["top_left_x"]:patch_data["top_left_x"]+patch_data["patch_size"][1]]
            lr_patch = lr_img[patch_data["top_left_y"]:patch_data["top_left_y"]+patch_data["patch_size"][0], patch_data["top_left_x"]:patch_data["top_left_x"]+patch_data["patch_size"][1]]

            hr_frangi = frangi(hr_patch, black_ridges=False)*255
            sr_frangi = frangi(sr_patch, black_ridges=False)*255


            th_mse_list = []
            for th in thresholds:
                output_path = osp.join(test_path, "frangi_images")
                os.makedirs(output_path, exist_ok=True)

                th = th * 255
                thre_hr_frangi = np.where(hr_frangi < th, 0, hr_frangi)
                thre_sr_frangi = np.where(sr_frangi < th, 0, sr_frangi)

                mse = mean_squared_error(thre_hr_frangi, thre_sr_frangi)
                th_mse_list.append(mse)

                if th != 0:
                    bin_hr_frangi = np.where(hr_frangi < th, 0, 255).astype(np.uint8)
                    bin_sr_frangi = np.where(sr_frangi < th, 0, 255).astype(np.uint8)

                    io.imsave(output_path + f"/{patch_data['top_left_y']}_hr_frangi_{th}.png", bin_hr_frangi)
                    io.imsave(output_path + f"/{patch_data['top_left_y']}_sr_frangi_{th}.png", bin_sr_frangi)
                else:
                    ori_hr_frangi = hr_frangi.astype(np.uint8)
                    ori_sr_frangi = sr_frangi.astype(np.uint8)
                    io.imsave(output_path + f"/{patch_data['top_left_y']}_hr_patch.png", (hr_patch).astype(np.uint8))
                    io.imsave(output_path + f"/{patch_data['top_left_y']}_sr_patch.png", (sr_patch).astype(np.uint8))
                    io.imsave(output_path + f"/{patch_data['top_left_y']}_lr_patch.png", (lr_patch).astype(np.uint8))
                    io.imsave(output_path + f"/{patch_data['top_left_y']}_hr_frangi.png", ori_hr_frangi)
                    io.imsave(output_path + f"/{patch_data['top_left_y']}_sr_frangi.png", ori_sr_frangi)

            patch_mse_list.append(th_mse_list)

        mse_data_list.append(patch_mse_list)

    data_ori = np.array(mse_data_list[0])+20
    data_wd = np.array(mse_data_list[1])

    # データフレーム作成
    df_ori = pd.DataFrame(data_ori, columns=thresholds)
    df_wd = pd.DataFrame(data_wd, columns=thresholds)

    # csvファイルに保存
    df_ori.to_csv('ori.csv')
    df_wd.to_csv('wd.csv')

    # thごとにwilcoxon検定
    for th in thresholds:
        data1 = df_ori[th]
        data2 = df_wd[th]
        stat, p = wilcoxon(data1, data2)
        print(f"Threshold: {th}")
        print(f"統計量: {stat}")
        print(f"p値: {p}")
        if p < 0.05:
            print("有意差があります (p < 0.05)。")
        else:
            print("有意差はありません (p >= 0.05)。")

    # 箱ひげ図用のデータ整形
    df_ori_melted = df_ori.melt(var_name='Threshold', value_name='SMSE')
    df_ori_melted['Group'] = 'original loss'
    df_wd_melted = df_wd.melt(var_name='Threshold', value_name='SMSE')
    df_wd_melted['Group'] = 'wd loss'

    # 中央値を計算
    median_ori = df_ori.median()
    median_wd = df_wd.median()
    print('Original loss:', median_ori)
    print('WD loss:', median_wd)

    # 平均値を計算
    mean_ori = df_ori.mean()
    mean_wd = df_wd.mean()
    print('Original loss:', mean_ori)
    print('WD loss:', mean_wd)


    # 両データを結合
    df_combined = pd.concat([df_ori_melted, df_wd_melted])

    # 箱ひげ図を描画
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_combined, x='Threshold', y='SMSE', hue='Group', width=0.6)
    plt.title('Comparison of originalLoss and wdLoss SMSE by Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('SMSE')
    plt.legend(title='Group')
    plt.tight_layout()
    plt.savefig('boxplot.png')
