from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import SimpleITK as sitk
import os.path as osp
from glob import glob


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

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
            if split == 'val': #ここはval
                hr_files = glob(osp.join(dataroot+'/hr_{}/*'.format(r_resolution)))
                lr_files = glob(osp.join(dataroot+'/lr_{}/*'.format(l_resolution)))
                sr_files = glob(osp.join(dataroot+'/sr_{}_{}/*'.format(l_resolution, r_resolution)))
                self.sr_path = [Util.get_paths_from_mhds(
                    '{}'.format(file)) for file in sr_files]
                self.hr_path = [Util.get_paths_from_mhds(
                    '{}'.format(file)) for file in hr_files]
                if self.need_LR:
                    self.lr_path = [Util.get_paths_from_mhds(
                    '{}'.format(file)) for file in lr_files]
                self.dataset_len = len(self.hr_path)
                if self.data_len <= 0:
                    self.data_len = self.dataset_len
                else:
                    self.data_len = min(self.data_len, self.dataset_len)
            elif split == 'train':#ここはtrain
                self.sr_path = Util.get_paths_from_mhds(
                    '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
                self.hr_path = Util.get_paths_from_mhds(
                    '{}/hr_{}'.format(dataroot, r_resolution))
                if self.need_LR:
                    self.lr_path = Util.get_paths_from_mhds(
                        '{}/lr_{}'.format(dataroot, l_resolution))
                self.dataset_len = len(self.hr_path)
                if self.data_len <= 0:
                    self.data_len = self.dataset_len
                else:
                    self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None
        # print(index)

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
            if self.split == 'val': #validationはpatchの配列
                imgs_HR = [sitk.ReadImage(file) for file in self.hr_path[index]]
                imgs_SR = [sitk.ReadImage(file) for file in self.sr_path[index]]
                nda_imgs_HR = [sitk.GetArrayFromImage(img_HR) for img_HR in imgs_HR]
                nda_imgs_SR = [sitk.GetArrayFromImage(img_SR) for img_SR in imgs_SR]
                img_HR = [Image.fromarray(nda_img_HR) for nda_img_HR in nda_imgs_HR]
                img_SR = [Image.fromarray(nda_img_SR) for nda_img_SR in nda_imgs_SR]
                if self.need_LR:
                    imgs_LR = [sitk.ReadImage(file) for file in self.lr_path[index]]
                    nda_imgs_LR = [sitk.GetArrayFromImage(img_LR) for img_LR in imgs_LR]
                    img_LR = [Image.fromarray(nda_img_LR) for nda_img_LR in nda_imgs_LR]
            else:
                img_HR = sitk.ReadImage(self.hr_path[index])
                img_SR = sitk.ReadImage(self.sr_path[index])
                nda_img_HR = sitk.GetArrayFromImage(img_HR)
                nda_img_SR = sitk.GetArrayFromImage(img_SR)
                img_HR = Image.fromarray(nda_img_HR)
                img_SR = Image.fromarray(nda_img_SR)
                if self.need_LR:
                    img_LR = sitk.ReadImage(self.lr_path[index])
                    nda_img_LR = sitk.GetArrayFromImage(img_LR)
                    img_LR = Image.fromarray(nda_img_LR)
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(0, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(0, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
