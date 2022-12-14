'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data
import os.path as osp
from glob import glob


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
    files = glob(osp.join(dataset_opt['dataroot']+'/hr_{}/*'.format(dataset_opt['r_resolution']))) #スライスファイルの名前取得
    files = [file.split('/')[4] for file in files]
    from data.LRHR_dataset import LRHRDataset as D
    if phase == 'train':
        dataset = D(dataroot=dataset_opt['dataroot'],
                    datatype=dataset_opt['datatype'],
                    l_resolution=dataset_opt['l_resolution'],
                    r_resolution=dataset_opt['r_resolution'],
                    split=phase,
                    data_len=dataset_opt['data_length'],
                    need_LR=(mode == 'LRHR'),
                    slice_file=0
                    )
    elif phase == 'val':
        dataset = [D(dataroot=dataset_opt['dataroot'],
                    datatype=dataset_opt['datatype'],
                    l_resolution=dataset_opt['l_resolution'],
                    r_resolution=dataset_opt['r_resolution'],
                    split=phase,
                    data_len=dataset_opt['data_length'],
                    need_LR=(mode == 'LRHR'),
                    slice_file=file
                    ) for file in files]
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
