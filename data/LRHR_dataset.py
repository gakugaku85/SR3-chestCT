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


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False, slice_file=0):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        self.hr_paths = []
        self.lr_paths = [] 
        self.sr_paths = []

        self.hr_imgs = []
        self.lr_imgs = [] 
        self.sr_imgs = []
        if datatype == 'mhd':
            if split == 'val':
                self.sr_path = Util.get_paths_from_mhds(
                    '{}/sr_{}_{}/{}'.format(dataroot, l_resolution, r_resolution, slice_file))
                self.hr_path = Util.get_paths_from_mhds(
                    '{}/hr_{}/{}'.format(dataroot, r_resolution, slice_file))
                self.dataset_len = len(self.hr_path)
                if self.data_len <= 0:
                    self.data_len = self.dataset_len
                else:
                    self.data_len = min(self.data_len, self.dataset_len)
            else:#ここはtrain
                for dataroot_ in dataroot.split():
                    self.sr_paths.append(Util.get_paths_from_mhds(
                        '{}/nonzero/sr_0_0'.format(dataroot_)))
                    self.hr_paths.append(Util.get_paths_from_mhds(
                        '{}/nonzero/hr_0'.format(dataroot_)))
                self.hr_path = list(itertools.chain.from_iterable(self.hr_paths))
                self.sr_path = list(itertools.chain.from_iterable(self.sr_paths))
                self.dataset_len = len(self.hr_path)
                if self.data_len <= 0:
                    self.data_len = self.dataset_len
                else:
                    self.data_len = min(self.data_len, self.dataset_len)
                    self.hr_path = self.hr_path[:self.data_len]
                    self.sr_path = self.sr_path[:self.data_len]
                print("slice_length : {}".format(self.data_len))
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))
        for hr, sr in tqdm(zip(self.hr_path, self.sr_path), desc='create datasets', total=len(self.hr_path)):
            img_HR = sitk.ReadImage(hr)
            img_SR = sitk.ReadImage(sr)
            nda_img_HR = sitk.GetArrayFromImage(img_HR)
            nda_img_SR = sitk.GetArrayFromImage(img_SR)
            self.hr_imgs.append(nda_img_HR)
            self.sr_imgs.append(nda_img_SR)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None
        GT_size = self.r_res

        if self.datatype == 'mhd':
            nda_img_HR = self.hr_imgs[index]
            nda_img_SR = self.sr_imgs[index]
            # if self.need_LR:
            #     img_LR = sitk.ReadImage(self.lr_path[index])
            #     nda_img_LR = sitk.GetArrayFromImage(img_LR)
            #     img_LR = Image.fromarray(nda_img_LR)
            if self.split == 'train':
                H, W = nda_img_HR.shape
                crop_h = 0 
                crop_w = 0
                while(1):
                    crop_h, crop_w = choose_lung_crop(H, W, GT_size)
                    img_HR = nda_img_HR[crop_h: crop_h + GT_size, crop_w : crop_w + GT_size]
                    # print(np.count_nonzero(img_HR==0)/np.count_nonzero(img_HR>=0))
                    if (np.count_nonzero(img_HR==0)/np.count_nonzero(img_HR>=0) <= 0.8):#黒の割合
                        break
                nda_img_HR = nda_img_HR[crop_h: crop_h + GT_size, crop_w : crop_w + GT_size]
                nda_img_SR = nda_img_SR[crop_h: crop_h + GT_size, crop_w : crop_w + GT_size]
            img_HR = Image.fromarray(nda_img_HR)
            img_SR = Image.fromarray(nda_img_SR)

        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(0, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(0, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}

def choose_lung_crop(H, W, GT_size):
    h = random.randint(0, H-GT_size)
    w = random.randint(0, W-GT_size)
    return h, w
