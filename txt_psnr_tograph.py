import re
import matplotlib.pyplot as plt
import os

# テキストファイルのパス
# file_path = "/take/gaku/SR3/SR3-chestCT/experiments/sr_patch_64_241125_090218/logs/val.log"# あなたのtxtファイルのパスに置き換えてください
file_path = '/take/gaku/SR3/SR3/experiments/past/sr_patch_64_230203_072414/logs/val.log'

# データを読み込む
with open(file_path, "r") as file:
    lines = file.readlines()

# iterとpsnrを抽出
iters = []
psnrs = []
ssims = []

for line in lines:
    match = re.search(r"iter:\s+([\d,]+).*?psnr:\s+([\d\.e\+\-]+).*?ssim:\s+([\d\.e\+\-]+)", line)
    if match:
        iter_value = int(match.group(1).replace(",", ""))  # カンマを削除して数値に変換
        psnr_value = float(match.group(2))
        ssim_value = float(match.group(3))
        iters.append(iter_value)
        psnrs.append(psnr_value)
        ssims.append(ssim_value)

# 一番PSNRが高い点を探す
max_psnr = max(psnrs)
max_index = psnrs.index(max_psnr)  # インデックスを取得
max_iter = iters[max_index]  # インデックスからiterの値を取得

# プロット
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# PSNRのグラフ
ax1.plot(iters, psnrs, label="PSNR")
ax1.scatter(max_iter, max_psnr, color='red')  # 一番高いPSNRの点を赤で打つ
ax1.set_xlabel("Iteration", fontsize=12)
ax1.set_ylabel("PSNR", fontsize=12)
ax1.set_title("PSNR vs Iteration", fontsize=14)
ax1.legend(fontsize=12)

# SSIMのグラフ
ax2.plot(iters, ssims, label="SSIM", color='orange')
ax2.scatter(max_iter, ssims[max_index], color='red')
ax2.set_xlabel("Iteration", fontsize=12)
ax2.set_ylabel("SSIM", fontsize=12)
ax2.set_title("SSIM vs Iteration", fontsize=14)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(file_path), "psnr_ssim_vs_iter.png"))