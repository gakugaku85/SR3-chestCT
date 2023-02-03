import numpy as np
import SimpleITK as sitk
from glob import glob
from natsort import natsorted
import os.path as osp

def zero_padding(image, new_shape):
    h, w = image.shape
    padded_image = np.zeros(new_shape)
    padded_image[:h, :w] = image
    return padded_image

paths = natsorted(glob(osp.join('experiments/sr_sam_dan/results/*mhd')))

for i, path in enumerate(paths):
    image = sitk.GetArrayFromImage(sitk.ReadImage(path))
    new_shape = (1713, 1948)
    padded_image = zero_padding(image, new_shape)
    padded_image = sitk.GetImageFromArray(padded_image)
    sitk.WriteImage(padded_image, 'experiments/sr_sam_dan/my_results/{}_sr.mhd'.format(i))