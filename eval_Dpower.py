import os
import os.path as osp
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from icecream import ic
from natsort import natsorted
from scipy.fftpack import fft2, fftshift


def create_dpower_graph(path):
    dpower_list = []
    for dir_path, _, fnames in natsorted(os.walk(path)):
        dpower_list_val = []
        ite = 0
        for fname in natsorted(fnames):
            if fname.endswith("hr.mhd"):
                ite = fname.split("_")[0]
                ic(ite)
                if int(ite) < 38000:
                    continue

                mhd_file_path = os.path.join(dir_path, fname)
                hr_image = sitk.GetArrayFromImage(sitk.ReadImage(mhd_file_path))
                sr_image = sitk.GetArrayFromImage(sitk.ReadImage(mhd_file_path.replace("hr", "sr")))

                # centerから256x256を切り出す
                h, w = hr_image.shape
                ch, cw = 256, 256
                start_h = (h - ch) // 2
                start_w = (w - cw) // 2
                hr_image = hr_image[start_h:start_h + ch, start_w:start_w + cw]
                sr_image = sr_image[start_h:start_h + ch, start_w:start_w + cw]

                dpower_list_val.append(calculate_dpower(hr_image, sr_image))
        dpow_low_mean = np.mean([dp["Dpower_band_0-1"] for dp in dpower_list_val])
        dpow_high_mean = np.mean([dp["Dpower_band_1-2"] for dp in dpower_list_val])
        dpow_sohigh_mean = np.mean([dp["Dpower_band_2-3"] for dp in dpower_list_val])
        dpow_sosohigh_mean = np.mean([dp["Dpower_band_3-4"] for dp in dpower_list_val])
        dpower_list.append({"ite": ite, "dpow_low_mean": dpow_low_mean, "dpow_high_mean": dpow_high_mean, "dpow_sohigh_mean": dpow_sohigh_mean, "dpow_sosohigh_mean": dpow_sosohigh_mean})

    dpow_low_mean_list = [dp["dpow_low_mean"] for dp in dpower_list]
    dpow_high_mean_list = [dp["dpow_high_mean"] for dp in dpower_list]
    dpow_sohigh_mean_list = [dp["dpow_sohigh_mean"] for dp in dpower_list]
    dpow_sosohigh_mean_list = [dp["dpow_sosohigh_mean"] for dp in dpower_list]
    ite_list = [dp["ite"] for dp in dpower_list]

        # サブプロット作成
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()  # 2次元を1次元のリストに変換

    ite_scaled = [float(i) / 1e3 for i in ite_list]

    iter_max = 96
    max_index = ite_scaled.index(iter_max)

    # 各サブプロットにデータをプロット
    axes[0].plot(ite_scaled, dpow_low_mean_list, label="Dpower_band_0-1")
    axes[0].set_title("Dpower Band 0-1")
    axes[0].scatter(ite_scaled[max_index], dpow_low_mean_list[max_index], color='red')
    axes[0].set_xlabel("iteration (x1e3)")  # 軸ラベルにスケールを明記
    axes[0].set_ylabel("Dpower")
    axes[0].legend()

    axes[1].plot(ite_scaled, dpow_high_mean_list, label="Dpower_band_1-2", color="orange")
    axes[1].set_title("Dpower Band 1-2")
    axes[1].scatter(ite_scaled[max_index], dpow_high_mean_list[max_index], color='red')
    axes[1].set_xlabel("iteration (x1e3)")
    axes[1].set_ylabel("Dpower")
    axes[1].legend()

    axes[2].plot(ite_scaled, dpow_sohigh_mean_list, label="Dpower_band_2-3", color="green")
    axes[2].set_title("Dpower Band 2-3")
    axes[2].scatter(ite_scaled[max_index], dpow_sohigh_mean_list[max_index], color='red')
    axes[2].set_xlabel("iteration (x1e3)")
    axes[2].set_ylabel("Dpower")
    axes[2].legend()

    axes[3].plot(ite_scaled, dpow_sosohigh_mean_list, label="Dpower_band_3-4", color="red")
    axes[3].set_title("Dpower Band 3-4")
    axes[3].scatter(ite_scaled[max_index], dpow_sosohigh_mean_list[max_index], color='red')
    axes[3].set_xlabel("iteration (x1e3)")
    axes[3].set_ylabel("Dpower")
    axes[3].legend()

    # レイアウト調整と保存
    fig.tight_layout()
    plt.savefig(os.path.join(path, "dpower.png"))

def calculate_dpower(hr, super_res, scale_factor=4):
    """
    Calculate Dpower for frequency band analysis, dividing Nyquist frequency into equal parts.

    Args:
        hr (numpy.ndarray): High-resolution ground truth image.
        super_res (numpy.ndarray): Super-resolved image.
        scale_factor (int): Super-resolution scaling factor (e.g., 4 for 4x SR).

    Returns:
        dict: Dictionary with Dpower for each frequency band.
    """
    # Ensure images are grayscale
    if len(hr.shape) == 3:
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY)
    if len(super_res.shape) == 3:
        super_res = cv2.cvtColor(super_res, cv2.COLOR_BGR2GRAY)

    # Resize images to the same size if necessary
    if hr.shape != super_res.shape:
        super_res = cv2.resize(super_res, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Compute Fourier transforms
    fft_hr = np.fft.fftshift(np.fft.fftn(hr))
    fft_sr = np.fft.fftshift(np.fft.fftn(super_res))

    # Compute power spectra
    power_hr = np.abs(fft_hr) ** 2
    power_sr = np.abs(fft_sr) ** 2

    # Define image dimensions
    h, w = hr.shape
    center_x, center_y = w // 2, h // 2

    # Calculate maximum radius (Nyquist frequency radius)
    nyquist_radius = min(h, w) // 2
    band_width = nyquist_radius / scale_factor  # Divide Nyquist frequency into `scale_factor` bands

    # Initialize results dictionary
    dpower_results = {}

    # Create frequency bands and calculate Dpower for each band
    y, x = np.ogrid[:h, :w]
    distance_from_center = np.maximum(np.abs(x - center_x), np.abs(y - center_y))

    for i in range(scale_factor):
        # Define band mask
        lower_bound = i * band_width
        upper_bound = (i + 1) * band_width
        band_mask = (distance_from_center > lower_bound) & (distance_from_center <= upper_bound)

        # Calculate Dpower for the current band
        band_diff = np.mean(np.abs(power_hr[band_mask] - power_sr[band_mask]))

        dpower_results[f"Dpower_band_{i}-{i + 1}"] = band_diff

    return dpower_results

def cal_test_Dpower(base_path):
    hr_paths = natsorted(glob(osp.join(base_path, '*_hr.mhd')))
    sr_paths = natsorted(glob(osp.join(base_path, '*_sr.mhd')))
    dpower_results = []
    for hr_path, sr_path in zip(hr_paths, sr_paths):
        hr_image = sitk.GetArrayFromImage(sitk.ReadImage(hr_path))
        sr_image = sitk.GetArrayFromImage(sitk.ReadImage(sr_path))

        # centerから256x256を切り出す
        h, w = hr_image.shape
        ch, cw = 256, 256
        start_h = (h - ch) // 2
        start_w = (w - cw) // 2
        hr_image = hr_image[start_h:start_h + ch, start_w:start_w + cw]
        sr_image = sr_image[start_h:start_h + ch, start_w:start_w + cw]

        dpower_result = calculate_dpower(hr_image, sr_image)
        dpower_results.append(dpower_result)
        # print(dpower_result)

    dpow_low_mean = np.mean([dp["Dpower_band_0-1"] for dp in dpower_results])
    dpow_high_mean = np.mean([dp["Dpower_band_1-2"] for dp in dpower_results])
    dpow_sohigh_mean = np.mean([dp["Dpower_band_2-3"] for dp in dpower_results])
    dpow_sosohigh_mean = np.mean([dp["Dpower_band_3-4"] for dp in dpower_results])
    print(f"dpow_low_mean: {dpow_low_mean:.2e}")
    print(f"dpow_high_mean: {dpow_high_mean:.2e}")
    print(f"dpow_sohigh_mean: {dpow_sohigh_mean:.2e}")
    print(f"dpow_sosohigh_mean: {dpow_sosohigh_mean:.2e}")

if __name__ == "__main__":
    base_path = '/take/gaku/SR3/SR3-chestCT/experiments/ori_2_27000-/results'
    # base_path = '/take/gaku/SR3/SR3-chestCT/experiments/val_ori_best/results'
    # cal_test_Dpower(base_path)
    # base_path = '/take/gaku/SR3/SR3-chestCT/experiments/val_wd_10_best/results'
    # cal_test_Dpower(base_path)
    create_dpower_graph(base_path)