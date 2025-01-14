import argparse
from tqdm import tqdm
import os
from skimage.filters import frangi
import os.path as osp
import numpy as np
import SimpleITK as sitk
from glob import glob
from icecream import ic
from multiprocessing import Pool, cpu_count

def process_file(args):
    file, out = args
    img = sitk.ReadImage(file)
    img = sitk.GetArrayFromImage(img)
    # ic(img.max(), img.min())
    img = (img - img.min()) / (img.max() - img.min())
    img = frangi(img, black_ridges=False)
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, osp.join(out, osp.basename(file)))

def create_frangi_img(path, out, num_workers):
    if not os.path.exists(out):
        os.makedirs(out)

    files = glob(osp.join(path, "*mhd"))

    # Create arguments for each file
    tasks = [(file, out) for file in files]

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_file, tasks), total=len(files)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='/take/dataset/microCT_slices_1784/nonzero/hr')
    parser.add_argument('--workers', '-w', type=int, default=cpu_count(),
                        help="Number of parallel workers (default: all available cores)")

    args = parser.parse_args()

    ic(args)

    out_path = args.path.replace("hr", "frangi")

    create_frangi_img(args.path, out_path, args.workers)
