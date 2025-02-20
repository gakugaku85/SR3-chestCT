import argparse
import json
import os
import os.path as osp
import random
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from icecream import ic
from natsort import natsorted
from PIL import Image
from scipy import ndimage
from skimage.filters import frangi
from skimage.measure import label, regionprops

import core.metrics as Metrics


def load_mhd(file_path):
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)  # SimpleITKイメージをNumPy配列に変換
    return array, image

def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def save_as_png(hr_patch, i, result_path, img_name):
    hr_patch = hr_patch.astype(np.uint8)
    hr_patch = Image.fromarray(hr_patch)
    hr_patch.save(osp.join(result_path, "{}_{}.png".format(img_name, i)))

def create_color_label_image(labels, nlabels, shape, output_path, img_name):
    img = np.zeros((*shape, 3), dtype=np.uint8)
    cols = []
    for j in range(1, nlabels):
        cols.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))
    for j in range(1, nlabels):
        if np.any(labels == j):
            img[labels == j, ] = cols[j - 1]
    cv2.imwrite(output_path + "ori_imgs/" + img_name + "label" + ".png", img)

def hysteresis_process(array, th, output_path, img_name):
    high_label_array = cv2.adaptiveThreshold(
        array, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=5, C=-5
    )
    nlabels, high_labels, stats, _ = cv2.connectedComponentsWithStats(high_label_array, connectivity=8)

    for i in range(1, nlabels):
        if stats[i, cv2.CC_STAT_AREA] <= 50:
            high_labels[high_labels == i] = 0
    high_label_array = (high_labels > 0).astype(np.uint8)

    create_color_label_image(high_labels, nlabels, array.shape, output_path, img_name=img_name+"_high_")

    low_label_array = cv2.adaptiveThreshold(
        array, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=161, C=-5
    )
    nlabels, low_labels, stats, _ = cv2.connectedComponentsWithStats(low_label_array, connectivity=8)

    for i in range(1, nlabels):  # 0は背景なのでスキップ
        if stats[i, cv2.CC_STAT_AREA] <= th:  # 面積がth以下の場合
            low_labels[low_labels == i] = 0

    create_color_label_image(low_labels, nlabels, array.shape, output_path, img_name=img_name+"_low_")

    hysteresis_array =  low_labels * high_label_array
    hysteresis_labels_unique = np.unique(hysteresis_array)[1:]
    hysteresis_labels_num = len(hysteresis_labels_unique)
    # ic(hysteresis_labels_num)

    overleft_hy_label = np.zeros_like(hysteresis_array).astype(np.uint8)
    for ilabel in hysteresis_labels_unique:
        overleft_hy_label[low_labels == ilabel] = ilabel

    return overleft_hy_label

def apply_filter_to_binary(input_img, output_path, th=20, img_name=""):
    array = input_img

    array = frangi(array, black_ridges=False)
    array = array*255
    array = array.astype(np.uint8)
    # ic(array.max(), array.min())

    binary_array = hysteresis_process(array, th, output_path, img_name) # ヒステリシス処理

    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_array, connectivity=8)

    for i in range(1, nlabels):  # 0は背景なのでスキップ
        if stats[i, cv2.CC_STAT_AREA] <= th:  # 面積がth以下の場合
            labels[labels == i] = 0

    unique_labels = np.unique(labels)[1:]
    nlabels_actual = len(unique_labels)
    # ic(nlabels_actual)

    create_color_label_image(labels, nlabels, binary_array.shape, output_path, img_name)

    binary_array = (labels > 0).astype(np.uint8)
    binary_image = sitk.GetImageFromArray(binary_array.astype(np.uint8))
    sitk.WriteImage(binary_image, output_path + img_name + "_binary.mhd")

    return binary_array, labels, nlabels_actual, stats

def count_tpcc_from_base_image(targer_labels, base_binary_label, result_path, i, img_name=""):
    shape = targer_labels.shape
    mix_label = targer_labels * base_binary_label
    new_labels = np.unique(mix_label)[1:]
    tpcc_num = len(new_labels)
    overleft_label = np.zeros_like(mix_label)
    for ilabel in new_labels:
        overleft_label[targer_labels == ilabel] = ilabel

    create_color_label_image(overleft_label, new_labels.max(), shape, result_path, "overleft_{}_{}".format(img_name, i))

    return tpcc_num

def count_tpcc_per_connected_from_base_image(target_labels, base_labels, result_path, i, img_name=""):
    shape = target_labels.shape

    # ステップ1: base_labels から二値化マスクを作成
    base_binary_label = (base_labels > 0).astype(np.uint8)

    # ステップ2: target_labels に二値化マスクを乗算して重なりのあるラベルのみを保持
    mix_labels = target_labels * base_binary_label

    # ステップ3: mix_labels から背景 (0) を除外してユニークなラベルを抽出
    new_labels = np.unique(mix_labels)
    new_labels = new_labels[new_labels > 0]  # 背景を除外

    # [i_label, base_label] のペアを格納するリストを初期化
    per_c_labels = []

    # ステップ4: new_labels の各ラベルについてループ
    for i_label in new_labels:
        # 現在のラベルのマスクを作成
        i_label_mask = (mix_labels == i_label)

        # i_label_mask と重なる base_labels のラベルを取得
        overlapping_base_labels = base_labels[i_label_mask]

        # 重なり領域からユニークなラベルを取得し、背景 (0) を除外
        unique_base_labels = np.unique(overlapping_base_labels)
        unique_base_labels = unique_base_labels[unique_base_labels > 0]

        # 重なりのある各 base_label と i_label のペアをリストに追加
        for base_label in unique_base_labels:
            per_c_labels.append([i_label, base_label])

    # per
    base_label_count = {}
    for i_label, base_label in per_c_labels:
        if base_label in base_label_count:
            base_label_count[base_label] += 1
        else:
            base_label_count[base_label] = 1
    # base_label_countのすべての要素から1を引く
    base_label_count = {k: v-1 for k, v in base_label_count.items()}
    ic(base_label_count)

    # base_label_countの値の合計がTPCCの数
    tpcc_num = sum(base_label_count.values())
    delta_tpcc_num = tpcc_num / len(base_label_count)
    ic(delta_tpcc_num)
    # create_color_label_image(overleft_label, new_labels.max(), shape, result_path, "overleft_{}_{}".format(img_name, i))

    return delta_tpcc_num

def connect_all_image(range_num):
    wd_10_path = "experiments/val_wd_10_best/results/TPCC/"

    for i in range(range_num):
        img_list = []
        img_list.append(cv2.imread(result_path + "ori_imgs/" + "hr_patch_{}.png".format(i), cv2.IMREAD_COLOR))
        img_list.append(cv2.imread(result_path + "ori_imgs/" + "overleft_hr_baseGT_{}label.png".format(i), cv2.IMREAD_COLOR))
        img_list.append(cv2.imread(result_path + "ori_imgs/" + "sr_patch_{}.png".format(i), cv2.IMREAD_COLOR))
        img_list.append(cv2.imread(result_path + "ori_imgs/" + "overleft_sr_baseGT_{}label.png".format(i), cv2.IMREAD_COLOR))
        img_list.append(cv2.imread(wd_10_path + "ori_imgs/" + "sr_patch_{}.png".format(i), cv2.IMREAD_COLOR))
        img_list.append(cv2.imread(wd_10_path + "ori_imgs/" + "overleft_sr_baseGT_{}label.png".format(i), cv2.IMREAD_COLOR))
        img_list.append(cv2.imread(result_path + "ori_imgs/" + "label_patch_{}.png".format(i), cv2.IMREAD_COLOR))
        concat_img = cv2.hconcat(img_list)
        cv2.imwrite(result_path + "ori_imgs/concat_{}.png".format(i), concat_img)

def delete_label_th(labels, stats, nlabels, th):
    for i in range(1, nlabels):  # 0は背景なのでスキップ
        if stats[i, cv2.CC_STAT_AREA] <= th:  # 面積がth以下の場合
            labels[labels == i] = 0

    unique_labels = np.unique(labels)[1:]
    nlabels_actual = len(unique_labels)

    return labels, nlabels_actual

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default="val_ori_best")
    parser.add_argument('-t', '--th', type=int, default=10)
    args = parser.parse_args()

    test_path = "experiments/" + args.path + "/results"
    result_path = "experiments/" + args.path + "/results/TPCC/"

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(result_path + "/ori_imgs", exist_ok=True)

    json_path = osp.join('/take/dataset/microCT_slices_1792_2/patch_metadata_128.json')
    patch_data_list = load_json(json_path)

    hr_paths = natsorted(glob(osp.join(test_path, '*_hr.mhd')))
    sr_paths = natsorted(glob(osp.join(test_path, '*_sr.mhd')))

    hr_imgs = [load_mhd(hr_path)[0] for hr_path in hr_paths]
    sr_imgs = [load_mhd(sr_path)[0] for sr_path in sr_paths]

    delta_tpcc_list = []
    fpcc_list = []
    GT_hr_delta_tpcc_list = []
    GT_sr_delta_tpcc_list = []

    gaku_label_path = "/take/dataset/microCT_slices_1792_2/interlobular_septum_mask/Segmentation_600_label_gaku_1.nrrd"
    gaku_label = load_mhd(gaku_label_path)[0]
    print(gaku_label.shape)
    # 次元数が合わないのでchannelを減らす
    gaku_label = gaku_label[0]

    df_list = []

    for i, patch_data in enumerate(patch_data_list):
        file_name = patch_data["file_path"].split("/")[-1]
        img_idx = int(file_name.split(".")[0])/100

        hr_img = hr_imgs[int(img_idx)-1]
        sr_img = sr_imgs[int(img_idx)-1]

        # hr_patch = hr_imgs[i]
        # sr_patch = sr_imgs[i]

        hr_patch = hr_img[patch_data["top_left_y"]:patch_data["top_left_y"]+patch_data["patch_size"][0], patch_data["top_left_x"]:patch_data["top_left_x"]+patch_data["patch_size"][1]]
        sr_patch = sr_img[patch_data["top_left_y"]:patch_data["top_left_y"]+patch_data["patch_size"][0], patch_data["top_left_x"]:patch_data["top_left_x"]+patch_data["patch_size"][1]]
        label_patch = gaku_label[patch_data["top_left_y"]:patch_data["top_left_y"]+patch_data["patch_size"][0], patch_data["top_left_x"]:patch_data["top_left_x"]+patch_data["patch_size"][1]]

        save_as_png(hr_patch, i, result_path+"ori_imgs/", img_name="hr_patch")
        save_as_png(sr_patch, i, result_path+"ori_imgs/", img_name="sr_patch")
        save_as_png(label_patch*255, i, result_path+"ori_imgs/", img_name="label_patch")

        GT_nlabels, GT_labels, _, _ = cv2.connectedComponentsWithStats(label_patch, connectivity=8)
        GT_tpcc_num = GT_nlabels - 1

        hr_patch_binary, hr_labels, hr_nlabel, hr_stats = apply_filter_to_binary(hr_patch, result_path, img_name="hr_patch_{}".format(i), th=args.th)
        sr_patch_binary, sr_labels, sr_nlabel, sr_stats = apply_filter_to_binary(sr_patch, result_path, img_name="sr_patch_{}".format(i), th=args.th)

        hr_baseGT_tpcc_num = count_tpcc_from_base_image(hr_labels, label_patch, result_path, i, img_name="hr_baseGT")-1
        sr_baseGT_tpcc_num = count_tpcc_from_base_image(sr_labels, label_patch, result_path, i, img_name="sr_baseGT")-1 #背景がひとつ多いので-1

        # hr_baseGT_tpcc_num = count_tpcc_per_connected_from_base_image(hr_labels, GT_labels, result_path, i, img_name="hr_baseGT")
        # sr_baseGT_tpcc_num = count_tpcc_per_connected_from_base_image(sr_labels, GT_labels, result_path, i, img_name="sr_baseGT")

        # hr_labels, hr_nlabel= delete_label_th(hr_labels, hr_stats, hr_nlabel, 30)
        # sr_labels, sr_nlabel = delete_label_th(sr_labels, sr_stats, sr_nlabel, 30)

        hr_label = (hr_labels > 0).astype(np.uint8)
        hr_tpcc_num = hr_nlabel

        sr_baseHR_tpcc_num = count_tpcc_from_base_image(sr_labels, hr_label, result_path, i, img_name="sr_baseHR")

        GT_hr_delta_tpcc = hr_baseGT_tpcc_num - GT_tpcc_num
        GT_sr_delta_tpcc = sr_baseGT_tpcc_num - GT_tpcc_num
        GT_hr_delta_tpcc_list.append(GT_hr_delta_tpcc)
        GT_sr_delta_tpcc_list.append(GT_sr_delta_tpcc)

        sr_hr_delta_tpcc = sr_baseHR_tpcc_num - hr_tpcc_num
        delta_tpcc_list.append(sr_hr_delta_tpcc)

        fpcc = sr_nlabel - sr_baseHR_tpcc_num
        fpcc_list.append(fpcc)

        ic(GT_hr_delta_tpcc, GT_sr_delta_tpcc)

        df_list.append([GT_tpcc_num, hr_tpcc_num, sr_nlabel, hr_baseGT_tpcc_num, sr_baseGT_tpcc_num, GT_hr_delta_tpcc, GT_sr_delta_tpcc, sr_baseHR_tpcc_num, sr_hr_delta_tpcc, fpcc])

    print("GT_hr_delta_tpcc: ", np.mean(GT_hr_delta_tpcc_list).round(2), "GT_sr_delta_tpcc: ", np.mean(GT_sr_delta_tpcc_list).round(2), "sr-hr_delta_tpcc: ", np.mean(delta_tpcc_list).round(2))
    df = pd.DataFrame(df_list, columns=["GT_tpcc_num", "hr_label_num", "sr_label_num", "hr_baseGT_tpcc_num", "sr_baseGT_tpcc_num", "GT_hr_delta_tpcc", "GT_sr_delta_tpcc", "sr_baseHR_tpcc_num", "sr_hr_delta_tpcc", "fpcc"])
    df.to_csv(result_path + args.path + "_result.csv", index=False)

    # i = 20
    connect_all_image(i+1)
