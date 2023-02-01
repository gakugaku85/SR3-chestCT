'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data
import os.path as osp
from glob import glob
from natsort import natsorted


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    files = glob(osp.join(dataset_opt['dataroot']+'/hr_0/*.mhd')) #スライスファイルの名前取得
    file_paths = natsorted([file.split('/')[4] for file in files])
    # print(files)
    from data.LRHR_dataset import LRHRDataset as D
    if phase == 'train':
        dataset = D(dataroot=dataset_opt['dataroot'],
                    datatype=dataset_opt['datatype'],
                    l_resolution=dataset_opt['l_resolution'],
                    hr_patch_size=dataset_opt['r_resolution'],
                    split=phase,
                    data_len=dataset_opt['data_length'],
                    need_LR=(mode == 'HR'),
                    black_ratio=dataset_opt['black_ratio']
                    )
    elif phase == 'val':
        dataset = [D(dataroot=dataset_opt['dataroot'],
                    datatype=dataset_opt['datatype'],
                    l_resolution=dataset_opt['l_resolution'],
                    hr_patch_size=dataset_opt['r_resolution'],
                    split=phase,
                    data_len=dataset_opt['data_length'],
                    need_LR=(mode == 'HR'),
                    overlap=dataset_opt['overlap'],
                    slice_file=path) for path in file_paths]
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
