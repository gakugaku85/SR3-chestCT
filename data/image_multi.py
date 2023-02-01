import cv2
from natsort import natsorted
from glob import glob
import os
import os.path as osp

path = '../dataset/mask_1794'
# 画像の読み込み
mask_files = natsorted(glob(osp.join(path+'_png_2/*.png')))
sobel_mask_files = natsorted(glob(osp.join(path+'_sobel_png_1_reverse/*.png')))
out_path = path + '_sobel_png_1_outside'

os.makedirs(out_path, exist_ok=True)
i = 0
for mask, sobel in zip(mask_files, sobel_mask_files):
    i += 1 
    mask = cv2.imread(mask)
    sobel = cv2.imread(sobel)

# 掛け算
    result = cv2.multiply(mask, sobel)

# 結果画像を保存
    cv2.imwrite(out_path+'/{}.png'.format(i), result)
