import os
import cv2
from natsort import natsorted
from scipy.io import loadmat
import SimpleITK as sitk
import numpy as np
import PIL.Image as Image


def get_2x_bicubic_downsample(input_path, output_path, index):
    img = Image.open(input_path).convert('L')
    img = np.array(img, dtype=np.uint8)

    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    print(output_path)

    img = Image.fromarray(img_resized).convert('RGB')
    img.save(output_path + ".png")

def main():
    import argparse

    prog = argparse.ArgumentParser()
    prog.add_argument('--input_dir', '-i', type=str,
                      default="../datasetCT/from_sam/1794_microCT/mask_png",
                      help='path to input image directory')
    prog.add_argument('--output_dir', '-o', type=str,
                      default="../dataset/mask_1784_png",
                      help='output directory path')

    args = prog.parse_args()
    mask_paths = []

    os.makedirs(args.output_dir, exist_ok=True)

    for mask_filename in natsorted(os.listdir(os.path.abspath(args.input_dir))):
        mask_paths.append(mask_filename)

    for i in range(len(mask_paths)):
        input_image_path = os.path.join(args.input_dir, mask_paths[i])
        output_image_path = os.path.join(args.output_dir, mask_paths[i])
        get_2x_bicubic_downsample(input_image_path, output_image_path, i)
    prog.exit(0)

if __name__ == '__main__':
    main()

