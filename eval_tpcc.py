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

def save_as_png(hr_patch, sr_patch, i, result_path):
    hr_patch = hr_patch.astype(np.uint8)
    sr_patch = sr_patch.astype(np.uint8)
    hr_patch = Image.fromarray(hr_patch)
    sr_patch = Image.fromarray(sr_patch)
    hr_patch.save(osp.join(result_path, "hr_patch_{}.png".format(i)))
    sr_patch.save(osp.join(result_path, "sr_patch_{}.png".format(i)))

def create_color_label_image(labels, nlabels, shape, output_path, img_name):
    img = np.zeros((*shape, 3), dtype=np.uint8)
    cols = []
    for j in range(1, nlabels):
        cols.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))
    for j in range(1, nlabels):
        if np.any(labels == j):
            img[labels == j, ] = cols[j - 1]
    cv2.imwrite(output_path + "ori_imgs/" + img_name + "label" + ".png", img)

def hysteresis_process(array, th):
    high_label_array = cv2.adaptiveThreshold(
        array, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=5, C=-5
    )
    high_label_array = (high_label_array > 0).astype(np.uint8)
    nlabels, high_labels, stats, _ = cv2.connectedComponentsWithStats(high_label_array, connectivity=8)

    for i in range(1, nlabels):
        if stats[i, cv2.CC_STAT_AREA] <= th:
            high_labels[high_labels == i] = 0

    high_label_array = (high_labels > 0).astype(np.uint8)

    low_label_array = cv2.adaptiveThreshold(
        array, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=5, C=0
    )
    low_label_array = (low_label_array > 0).astype(np.uint8)
    nlabels, low_labels, stats, _ = cv2.connectedComponentsWithStats(low_label_array, connectivity=8)

    for i in range(1, nlabels):  # 0は背景なのでスキップ
        if stats[i, cv2.CC_STAT_AREA] <= th:  # 面積がth以下の場合
            low_labels[low_labels == i] = 0

    hysteresis_array =  low_labels * high_label_array
    hysteresis_labels_unique = np.unique(hysteresis_array)[1:]
    hysteresis_labels_num = len(hysteresis_labels_unique)
    # ic(hysteresis_labels_num)

    overleft_hy_label = np.zeros_like(hysteresis_array)
    for ilabel in hysteresis_labels_unique:
        overleft_hy_label[low_labels == ilabel] = ilabel

    return overleft_hy_label

def apply_filter_to_binaly(input_img, output_path, th=30, img_name=""):
    array = input_img

    array = frangi(array, black_ridges=False)
    array = array*255
    array = array.astype(np.uint8)
    # ic(array.max(), array.min())

    binary_array = hysteresis_process(array, th)

    # binary_array = cv2.adaptiveThreshold(
    #     array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY, blockSize=5, C=-5
    # )

    binary_array = (binary_array > 0).astype(np.uint8)  # 二値化結果を0/1に変換

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

    return binary_array, labels, nlabels_actual

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default="sr_patch_64_val_250118_042926")
    args = parser.parse_args()

    test_path = "experiments/" + args.path + "/results"
    json_path = "/take/dataset/microCT_slices_1792_2/kakuheki_patch_256/"
    result_path = "experiments/" + args.path + "/results/TPCC/"

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(result_path + "/ori_imgs", exist_ok=True)

    json_path = osp.join(json_path, 'patch_metadata.json')
    patch_data_list = load_json(json_path)

    hr_paths = natsorted(glob(osp.join(test_path, '*_hr.mhd')))
    sr_paths = natsorted(glob(osp.join(test_path, '*_sr.mhd')))

    hr_imgs = [load_mhd(hr_path)[0] for hr_path in hr_paths]
    sr_imgs = [load_mhd(sr_path)[0] for sr_path in sr_paths]

    delta_tpcc_list = []
    fpcc_list = []

    for i, patch_data in enumerate(patch_data_list):
        file_name = patch_data["file_path"].split("/")[-1]
        img_idx = int(file_name.split(".")[0])/100

        hr_img = hr_imgs[int(img_idx)-1]
        sr_img = sr_imgs[int(img_idx)-1]

        hr_patch = hr_img[patch_data["top_left_y"]:patch_data["top_left_y"]+patch_data["patch_size"][0], patch_data["top_left_x"]:patch_data["top_left_x"]+patch_data["patch_size"][1]]
        sr_patch = sr_img[patch_data["top_left_y"]:patch_data["top_left_y"]+patch_data["patch_size"][0], patch_data["top_left_x"]:patch_data["top_left_x"]+patch_data["patch_size"][1]]

        save_as_png(hr_patch, sr_patch, i, result_path + "/ori_imgs")

        hr_patch_binary, hr_labels, hr_nlabel = apply_filter_to_binaly(hr_patch, result_path, img_name="hr_patch_{}".format(i))
        sr_patch_binary, sr_labels, sr_nlabel = apply_filter_to_binaly(sr_patch, result_path, img_name="sr_patch_{}".format(i))

        hr_label = (hr_labels > 0).astype(np.uint8)
        mix_label = sr_labels * hr_label

        new_labels = np.unique(mix_label)[1:]
        sr_tpcc_num = len(new_labels)
        overleft_label = np.zeros_like(mix_label)
        for ilabel in new_labels:
            overleft_label[sr_labels == ilabel] = ilabel

        create_color_label_image(overleft_label, new_labels.max(), sr_patch_binary.shape, result_path, "overleft_label_{}".format(i))

        delta_tpcc = sr_tpcc_num - hr_nlabel
        delta_tpcc_list.append(delta_tpcc)

        fpcc = sr_nlabel - sr_tpcc_num
        fpcc_list.append(fpcc)

        ic(hr_nlabel, sr_nlabel, sr_tpcc_num)
        ic(delta_tpcc, fpcc)

    ic(np.mean(delta_tpcc_list), np.mean(fpcc_list))

