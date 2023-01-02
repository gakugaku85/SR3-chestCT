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
        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'mhd':
            if split == 'val':
                self.sr_path = Util.get_paths_from_mhds(
                    '{}/sr_{}_{}/{}'.format(dataroot, l_resolution, r_resolution, slice_file))
                self.hr_path = Util.get_paths_from_mhds(
                    '{}/hr_{}/{}'.format(dataroot, r_resolution, slice_file))
                if self.need_LR:
                    self.lr_path = Util.get_paths_from_mhds(
                        '{}/lr_{}/{}'.format(dataroot, l_resolution, slice_file))
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
                    if self.need_LR:
                        self.lr_paths.append(Util.get_paths_from_mhds(
                            '{}/nonzero/lr_0'.format(dataroot_)))
                self.hr_path = list(itertools.chain.from_iterable(self.hr_paths))
                self.sr_path = list(itertools.chain.from_iterable(self.sr_paths))
                self.dataset_len = len(self.hr_path)
                if self.data_len <= 0:
                    self.data_len = self.dataset_len
                else:
                    self.data_len = min(self.data_len, self.dataset_len)
                print("slice_length : {}".format(self.dataset_len))
                if self.need_LR:
                    self.lr_path = list(itertools.chain.from_iterable(self.lr_paths))
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None
        GT_size = self.r_res

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        elif self.datatype == 'img':
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        elif self.datatype == 'mhd':
            img_HR = sitk.ReadImage(self.hr_path[index])
            img_SR = sitk.ReadImage(self.sr_path[index])
            nda_img_HR = sitk.GetArrayFromImage(img_HR)
            nda_img_SR = sitk.GetArrayFromImage(img_SR)
            if self.need_LR:
                img_LR = sitk.ReadImage(self.lr_path[index])
                nda_img_LR = sitk.GetArrayFromImage(img_LR)
                img_LR = Image.fromarray(nda_img_LR)
            if self.split == 'train':
                H, W = nda_img_HR.shape
                crop_h = 0 
                crop_w = 0
                while(1):
                    crop_h, crop_w = choose_lung_crop(H, W, GT_size)
                    img_HR = nda_img_HR[crop_h: crop_h + GT_size, crop_w : crop_w + GT_size]
                    # print(np.count_nonzero(img_HR==0)/np.count_nonzero(img_HR>=0))
                    if (np.count_nonzero(img_HR==0)/np.count_nonzero(img_HR>=0) <= 0.4):#黒の割合
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
