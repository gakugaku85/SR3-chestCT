import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
import os.path as osp
from pathlib import Path
import lmdb
import numpy as np
import time
import SimpleITK as sitk
from glob import glob


def resize_and_convert(img, size, resample):
    if(img.size != size):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img


def image_convert_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return buffer.getvalue()

def zero_padding(img, sizes, over_lap):
    H, W =img.shape
    while H % (sizes[1]-over_lap) != 0:
        H=H+1
    while W % (sizes[1]-over_lap) != 0:
        W=W+1
    pil_image = Image.fromarray(img)
    new_img = Image.new(pil_image.mode, (W+over_lap, H+over_lap), 0)
    new_img.paste(pil_image, (0, 0))
    return np.array(new_img)

def resize_multiple(img, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False):
    lr_img = resize_and_convert(img, sizes[0], resample)
    hr_img = resize_and_convert(img, sizes[1], resample)
    sr_img = resize_and_convert(lr_img, sizes[1], resample)

    if lmdb_save:
        lr_img = image_convert_bytes(lr_img)
        hr_img = image_convert_bytes(hr_img)
        sr_img = image_convert_bytes(sr_img)

    return [lr_img, hr_img, sr_img]

def resize_multiple_patch_save(img, out_path, sizes=(16, 128), mhd_sizes=(), resample=Image.BICUBIC, img_file="", lmdb_save=False):
    os.makedirs(out_path, exist_ok=True)
    Hs = sizes[1]
    Ls = sizes[0]
    if sizes[0] == 0:
        os.makedirs('{}/nonzero/lr_{}'.format(out_path, sizes[0]), exist_ok=True)
        os.makedirs('{}/nonzero/hr_{}'.format(out_path, sizes[1]), exist_ok=True)
        os.makedirs('{}/nonzero/sr_{}_{}'.format(out_path, sizes[0], sizes[1]), exist_ok=True)
        os.makedirs('{}/lr_{}'.format(out_path, sizes[0]), exist_ok=True)
        os.makedirs('{}/hr_{}'.format(out_path, sizes[1]), exist_ok=True)
        os.makedirs('{}/sr_{}_{}'.format(out_path, sizes[0], sizes[1]), exist_ok=True)
        pil_image = Image.fromarray(img)
        lr_img = resize_and_convert(pil_image, mhd_sizes[0], resample)
        hr_img = resize_and_convert(pil_image, mhd_sizes[1], resample)
        sr_img = resize_and_convert(lr_img, mhd_sizes[1], resample) 
        # print(np.count_nonzero(img==0)/np.count_nonzero(img>=0))
        ratio = np.count_nonzero(img==0)/np.count_nonzero(img>=0)
        if ratio < 0.9:
            save_mhd(lr_img, '{}/nonzero/lr_{}/{}.mhd'.format(out_path, sizes[0], img_file))
            save_mhd(hr_img, '{}/nonzero/hr_{}/{}.mhd'.format(out_path, sizes[1], img_file))
            save_mhd(sr_img, '{}/nonzero/sr_{}_{}/{}.mhd'.format(out_path, sizes[0], sizes[1], img_file))
        save_mhd(lr_img, '{}/lr_{}/{}.mhd'.format(out_path, sizes[0], img_file))
        save_mhd(hr_img, '{}/hr_{}/{}.mhd'.format(out_path, sizes[1], img_file))
        save_mhd(sr_img, '{}/sr_{}_{}/{}.mhd'.format(out_path, sizes[0], sizes[1], img_file))
    else:
        os.makedirs('{}/nonzero/hr_{}/{}'.format(out_path, sizes[1], img_file), exist_ok=True)
        os.makedirs('{}/nonzero/sr_{}_{}/{}'.format(out_path, sizes[0], sizes[1], img_file), exist_ok=True)
        os.makedirs('{}/hr_{}/{}'.format(out_path, sizes[1], img_file), exist_ok=True)
        os.makedirs('{}/sr_{}_{}/{}'.format(out_path, sizes[0], sizes[1], img_file), exist_ok=True)

        over_lap =  50
        img = zero_padding(img, sizes, over_lap)

        pil_image = Image.fromarray(img)
        
        lr_img = resize_and_convert(pil_image, (img.shape[0]//4, img.shape[1]//4), resample)
        pil_hr_img = resize_and_convert(pil_image, (img.shape[0], img.shape[1]), resample)
        pil_sr_img = resize_and_convert(lr_img, (img.shape[0], img.shape[1]), resample)

        hr_img = np.array(pil_hr_img)
        sr_img = np.array(pil_sr_img)

        H, W= img.shape
        print(img.shape)
        nH = int(H / (Hs-over_lap))
        nW = int(W / (Hs-over_lap))

        hr_imgs = [Image.fromarray(hr_img[(Hs*x) - (over_lap*x) :(Hs*(x+1))-(over_lap*x), (Hs*y)-(over_lap*y) : (Hs*(y+1))-(over_lap*y)]) for x in range(nH) for y in range(nW)]
        # sr_imgs = [Image.fromarray(sr_img[sizes[1]*x:sizes[1]*(x+1), sizes[1]*y:sizes[1]*(y+1)]) for x in range(nH) for y in range(nW)]

        for i, img in enumerate(hr_imgs):
            save_mhd(img, '{}/hr_{}/{}/{}.mhd'.format(out_path, sizes[1], img_file, i))
        # for i, img in enumerate(sr_imgs):
        #     save_mhd(img, '{}/sr_{}_{}/{}/{}.mhd'.format(out_path, sizes[0], sizes[1], img_file, i))

def resize_worker(img_file, sizes, resample, lmdb_save=False):
    img = Image.open(img_file)
    img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes, resample=resample, lmdb_save=lmdb_save)

    return img_file.name.split('.')[0], out

def resize_worker_mhd(img_file, sizes, resample, out_path, lmdb_save=False):
    img_mhd = sitk.ReadImage(img_file)
    nda_img_mhd = sitk.GetArrayFromImage(img_mhd)
    img_file = os.path.basename(img_file).split('.')[0]
    h , w = nda_img_mhd.shape
    LR_h, LR_w = h//4, w//4
    mhd_size = [(LR_h, LR_w), (h, w)]
    resize_multiple_patch_save(nda_img_mhd, out_path, sizes=sizes, mhd_sizes=mhd_size, resample=resample, img_file=img_file, lmdb_save=lmdb_save)

class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, env, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.sizes = sizes

        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value

def prepare_process_worker(wctx, file_subset):
    for file in file_subset:
        i, imgs = wctx.resize_fn(file)
        lr_img, hr_img, sr_img = imgs
        if not wctx.lmdb_save:
            lr_img.save(
                '{}/lr_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], i.zfill(5)))
            hr_img.save(
                '{}/hr_{}/{}.png'.format(wctx.out_path, wctx.sizes[1], i.zfill(5)))
            sr_img.save(
                '{}/sr_{}_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], wctx.sizes[1], i.zfill(5)))
        else:
            with wctx.env.begin(write=True) as txn:
                txn.put('lr_{}_{}'.format(
                    wctx.sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                txn.put('hr_{}_{}'.format(
                    wctx.sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                txn.put('sr_{}_{}_{}'.format(
                    wctx.sizes[0], wctx.sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
        curr_total = wctx.inc_get()
        if wctx.lmdb_save:
            with wctx.env.begin(write=True) as txn:
                txn.put('length'.encode('utf-8'), str(curr_total).encode('utf-8'))

def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True

def save_mhd(img, img_path):
    # if img.ndim == 3:
    #     img = img[:, :, 0]
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, img_path)

def prepare(img_path, out_path, n_worker, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False):
    resize_fn = partial(resize_worker_mhd, sizes=sizes, resample=resample, out_path=out_path, lmdb_save=lmdb_save)
    files = glob(osp.join(img_path, "*mhd"))
    for file in tqdm(files):
        resize_fn(file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='../datasetCT/from_sam/1794_microCT/upperandlower_99point95percentile_BicubicDSby2_denoise_2dslices_masked_normalised_1794')
    parser.add_argument('--out', '-o', type=str,
                        default='../dataset/microCT_slices_1794_overlap')

    parser.add_argument('--size', type=str, default='64, 256')
    parser.add_argument('--n_worker', type=int, default=1)
    parser.add_argument('--resample', type=str, default='bicubic')
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]

    args.out = '{}_{}_{}'.format(args.out, sizes[0], sizes[1])
    prepare(args.path, args.out, args.n_worker, sizes=sizes, resample=resample, lmdb_save=args.lmdb)
