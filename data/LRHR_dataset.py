from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import SimpleITK as sitk
import os.path as osp
from glob import glob
import numpy as np
import itertools
from tqdm import tqdm
import math


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, hr_patch_size=64, split='train', data_len=-1, need_LR=False, slice_file=0, black_ratio = 0.8, overlap=4):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = hr_patch_size
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.black_ratio = black_ratio

        hr_paths = []
        sr_paths = []

        self.hr_imgs = [] 
        self.sr_imgs = []

        if split == 'train':
            for dataroot_ in dataroot.split():
                sr_paths.append(Util.get_paths_from_mhds('{}/nonzero/sr_0_0'.format(dataroot_)))
                hr_paths.append(Util.get_paths_from_mhds('{}/nonzero/hr_0'.format(dataroot_)))
            hr_path = list(itertools.chain.from_iterable(hr_paths))
            sr_path = list(itertools.chain.from_iterable(sr_paths))
            self.dataset_len = len(hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
                hr_path = hr_path[:self.data_len]
                sr_path = sr_path[:self.data_len]
            
            for hr, sr in tqdm(zip(hr_path, sr_path), desc='create datasets', total=len(hr_path)):
                img_HR = sitk.ReadImage(hr)
                img_SR = sitk.ReadImage(sr)
                nda_img_HR = sitk.GetArrayFromImage(img_HR)
                nda_img_SR = sitk.GetArrayFromImage(img_SR)
                self.hr_imgs.append(nda_img_HR)
                self.sr_imgs.append(nda_img_SR)
            print("train_slice_length : {}".format(self.data_len))

        if split == 'val':
            sr_path = '{}/sr_0_0/{}'.format(dataroot, slice_file)
            hr_path = '{}/hr_0/{}'.format(dataroot, slice_file)

            img_HR = sitk.ReadImage(hr_path)
            img_SR = sitk.ReadImage(sr_path)
            nda_img_HR = sitk.GetArrayFromImage(img_HR)
            nda_img_SR = sitk.GetArrayFromImage(img_SR)

            img_size = nda_img_HR.shape
            STRIDE = hr_patch_size - overlap #60

            hr_img = self.zero_padding(nda_img_HR, hr_patch_size)
            sr_img = self.zero_padding(nda_img_SR, hr_patch_size)

            coor = [(x, y)
                    for x in range(0, img_size[0], STRIDE)
                    for y in range(0, img_size[1], STRIDE)]

            for x, y in coor:
                self.hr_imgs.append(hr_img[x:x+hr_patch_size, y:y+hr_patch_size])
                self.sr_imgs.append(sr_img[x:x+hr_patch_size, y:y+hr_patch_size])

            self.dataset_len = len(self.hr_imgs)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

            print(hr_path)

    def __len__(self):
        return self.data_len

    def zero_padding(self, img, patch_size):
        H, W =img.shape
        pil_image = Image.fromarray(img)
        new_img = Image.new(pil_image.mode, (W+patch_size, H+patch_size), 0)
        new_img.paste(pil_image, (0, 0))
        return np.array(new_img)

    def __getitem__(self, index):
        img_HR = None
        GT_size = self.r_res

        nda_img_HR = self.hr_imgs[index]
        nda_img_SR = self.sr_imgs[index]
        
        if self.split == 'train':
            H, W = nda_img_HR.shape
            crop_h = 0 
            crop_w = 0
            while(1):
                crop_h, crop_w = choose_lung_crop(H, W, GT_size)
                img_HR = nda_img_HR[crop_h: crop_h + GT_size, crop_w : crop_w + GT_size]
                if (np.count_nonzero(img_HR==0)/np.count_nonzero(img_HR>=0) <= self.black_ratio):#黒の割合
                    break
            nda_img_HR = nda_img_HR[crop_h: crop_h + GT_size, crop_w : crop_w + GT_size]
            nda_img_SR = nda_img_SR[crop_h: crop_h + GT_size, crop_w : crop_w + GT_size]

        img_HR = Image.fromarray(nda_img_HR)
        img_SR = Image.fromarray(nda_img_SR)

        [img_SR, img_HR] = Util.transform_augment([img_SR, img_HR], split=self.split, min_max=(0, 1))
        return {'HR': img_HR, 'SR': img_SR}

def choose_lung_crop(H, W, GT_size):
    h = random.randint(0, H-GT_size)
    w = random.randint(0, W-GT_size)
    return h, w
