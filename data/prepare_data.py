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
    if(img.size[0] != size):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img


def image_convert_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return buffer.getvalue()


def resize_multiple(img, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False):
    lr_img = resize_and_convert(img, sizes[0], resample)
    hr_img = resize_and_convert(img, sizes[1], resample)
    sr_img = resize_and_convert(lr_img, sizes[1], resample)

    if lmdb_save:
        lr_img = image_convert_bytes(lr_img)
        hr_img = image_convert_bytes(hr_img)
        sr_img = image_convert_bytes(sr_img)

    return [lr_img, hr_img, sr_img]

def resize_multiple_patch_save(img, out_path, sizes=(16, 128), resample=Image.BICUBIC, img_file="", lmdb_save=False):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs('{}/nonzero/lr_{}/{}'.format(out_path, sizes[0], img_file), exist_ok=True)
    os.makedirs('{}/nonzero/hr_{}/{}'.format(out_path, sizes[1], img_file), exist_ok=True)
    os.makedirs('{}/nonzero/sr_{}_{}/{}'.format(out_path, sizes[0], sizes[1], img_file), exist_ok=True)
    os.makedirs('{}/lr_{}/{}'.format(out_path, sizes[0], img_file), exist_ok=True)
    os.makedirs('{}/hr_{}/{}'.format(out_path, sizes[1], img_file), exist_ok=True)
    os.makedirs('{}/sr_{}_{}/{}'.format(out_path, sizes[0], sizes[1], img_file), exist_ok=True)

    H, W= img.shape
    nH = int(H/sizes[1])
    nW = int(W/sizes[1])
    
    imgs = [img[sizes[1]*x:sizes[1]*(x+1), sizes[1]*y:sizes[1]*(y+1)] for x in range(nH) for y in range(nW)]

    for i, img in enumerate(imgs):
        pil_image = Image.fromarray(img)
        lr_img = resize_and_convert(pil_image, sizes[0], resample)
        hr_img = resize_and_convert(pil_image, sizes[1], resample)
        sr_img = resize_and_convert(lr_img, sizes[1], resample)
        if np.all(img != 0):
            save_mhd(lr_img, '{}/nonzero/lr_{}/{}/{}.mhd'.format(out_path, sizes[0], img_file, i))
            save_mhd(hr_img, '{}/nonzero/hr_{}/{}/{}.mhd'.format(out_path, sizes[1], img_file, i))
            save_mhd(sr_img, '{}/nonzero/sr_{}_{}/{}/{}.mhd'.format(out_path, sizes[0], sizes[1], img_file, i))
        save_mhd(lr_img, '{}/lr_{}/{}/{}.mhd'.format(out_path, sizes[0], img_file, i))
        save_mhd(hr_img, '{}/hr_{}/{}/{}.mhd'.format(out_path, sizes[1], img_file, i))
        save_mhd(sr_img, '{}/sr_{}_{}/{}/{}.mhd'.format(out_path, sizes[0], sizes[1], img_file, i))

def resize_worker(img_file, sizes, resample, lmdb_save=False):
    img = Image.open(img_file)
    img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes, resample=resample, lmdb_save=lmdb_save)

    return img_file.name.split('.')[0], out

def resize_worker_mhd(img_file, sizes, resample, out_path, lmdb_save=False):
    img_mhd = sitk.ReadImage(img_file)
    nda_img_mhd = sitk.GetArrayFromImage(img_mhd)
    img_file = os.path.basename(img_file).split('.')[0]
    resize_multiple_patch_save(nda_img_mhd, out_path, sizes=sizes, resample=resample, img_file=img_file, lmdb_save=lmdb_save)

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
    # files = [p for p in Path('{}'.format(img_path)).glob(f'**/*')]
    files = glob(osp.join(img_path, "*mhd"))

    # if n_worker > 1:
    #     # prepare data subsets

    #     file_subsets = np.array_split(files, n_worker)
    #     worker_threads = []
    #     wctx = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, sizes)

    #     # start worker processes, monitor results
    #     for i in range(n_worker):
    #         proc = Process(target=prepare_process_worker, args=(wctx, file_subsets[i]))
    #         proc.start()
    #         worker_threads.append(proc)
        
    #     total_count = str(len(files))
    #     while not all_threads_inactive(worker_threads):
    #         print("\r{}/{} images processed".format(wctx.value(), total_count), end=" ")
    #         time.sleep(0.1)

    # else:
    total = 0
    for file in tqdm(files):
        resize_fn(file)
        total += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='../datasetCT/from_sam/1794_microCT/upperandlower_99point95percentile_BicubicDSby2_denoise_2dslices_masked_normalised_1794')
    parser.add_argument('--out', '-o', type=str,
                        default='./dataset/microCT_slices_1794')

    parser.add_argument('--size', type=str, default='16, 64')
    parser.add_argument('--n_worker', type=int, default=1)
    parser.add_argument('--resample', type=str, default='bicubic')
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]

    args.out = '{}_{}_{}'.format(args.out, sizes[0], sizes[1])
    prepare(args.path, args.out, args.n_worker,
            sizes=sizes, resample=resample, lmdb_save=args.lmdb)
