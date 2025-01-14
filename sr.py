import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
from tqdm import tqdm
from natsort import natsorted
import os
import numpy as np
import PIL.Image as Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr3_64_256.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', '-w', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval','-l', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("iter/*", step_metric="current_step")
        wandb.define_metric("validation/*", step_metric="current_step")
        val_step = 0
    else:
        wandb_logger = None

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'val':
            val_sets = Data.create_dataset(dataset_opt, phase)
            val_loaders = [Data.create_dataloader(val_set, dataset_opt, phase) for val_set in val_sets]
    logger.info('Initial Dataset Finished')

    mask_imgs = []
    for mask_filename in natsorted(os.listdir(os.path.abspath(opt['datasets']['val']['maskroot']))):
        mask_path = os.path.join(opt['datasets']['val']['maskroot'], mask_filename)
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask, dtype=np.uint8)
        mask_imgs.append(mask)
        # print(mask_path)
    logger.info('Initial mask Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])


    if opt['phase'] == 'train':
        result_train_path = '{}/train'.format(opt['path']['results'])
        os.makedirs(result_train_path, exist_ok=True)
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # recon_out = diffusion.print_train_result() #trainingの結果を出力
                # recon_img = Metrics.tensor2mhd(recon_out)  # uint8
                # Metrics.save_mhd(recon_img,'{}/{}_recon.mhd'.format(result_train_path, current_step))
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    if wandb_logger:
                        wandb_logger.log_metrics(logs)
                    logger.info(message)
                #train out
                if current_step % opt['train']['train_print_freq'] == 0 and current_step >= opt['train']['over_train_print']:
                    logger.info("<print_train_test>")
                    diffusion.test(continous=False)
                    visuals = diffusion.get_current_visuals()
                    b = visuals['HR'].shape[0]
                    train_out = torch.cat([visuals['INF'][b-1], visuals['SR'][b-1], visuals['HR'][b-1]], dim=2)
                    train_img = Metrics.tensor2mhd(train_out)  # uint8
                    Metrics.save_mhd(train_img, '{}/{}_train.mhd'.format(result_train_path, current_step))

                # validation
                if current_step % opt['train']['val_freq'] == 0 and current_step >= opt['train']['over_val']:
                    result_path = '{}/{}'.format(opt['path']['results'], current_step)
                    os.makedirs(result_path, exist_ok=True)
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    idx = 0

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')

                    for _,  val_loader in enumerate(val_loaders):
                        val_i = 0
                        psnr = 0.0
                        ssim = 0.0
                        val_psnr = 0.0
                        val_ssim = 0.0
                        sr_imgs = []
                        hr_imgs = []
                        fake_imgs = []
                        idx += 1
                        for _, val_data in enumerate(val_loader):
                            diffusion.feed_data(val_data)
                            diffusion.test(continous=False)
                            visuals = diffusion.get_current_visuals()
                            for i in range(visuals['SR'].shape[0]):
                                hr_patch = Metrics.tensor2mhd(visuals['HR'][i])
                                sr_patch = Metrics.tensor2mhd(visuals['SR'][i])
                                fake_patch = Metrics.tensor2mhd(visuals['INF'][i])
                                if val_i == opt['train']['val_i']:
                                    val_img = np.concatenate([fake_patch, sr_patch, hr_patch], axis=1) # uint8
                                    Metrics.save_mhd(val_img, '{}/{}_{}_val.mhd'.format(result_path, current_step, idx))
                                    val_psnr = Metrics.calculate_psnr(sr_patch, hr_patch)
                                    val_ssim = Metrics.calculate_ssim(sr_patch, hr_patch)
                                sr_imgs.append(sr_patch)  # uint8
                                hr_imgs.append(hr_patch)  # uint8
                                fake_imgs.append(fake_patch)  # uint8
                                val_i += 1
                        sr_img = Metrics.concatImage(sr_imgs, opt)
                        hr_img = Metrics.concatImage(hr_imgs, opt)
                        fake_img = Metrics.concatImage(fake_imgs, opt)
                        # save
                        Metrics.save_mhd(hr_img, '{}/{}_{}_hr.mhd'.format(result_path, current_step, idx))
                        Metrics.save_mhd(sr_img, '{}/{}_{}_sr.mhd'.format(result_path, current_step, idx))
                        Metrics.save_mhd(fake_img, '{}/{}_{}_inf.mhd'.format(result_path, current_step, idx))

                        mask = mask_imgs[idx-1]
                        psnr = Metrics.calculate_psnr_mask(sr_img, hr_img, mask)
                        ssim = Metrics.calculate_ssim_mask(sr_img, hr_img, mask)
                        logger.info('# Validation_patch{} # PSNR: {:.4e}, # SSIM: {:.4e}'.format(idx ,val_psnr, val_ssim))
                        logger.info('# Validation{} # PSNR: {:.4e}, # SSIM: {:.4e}'.format(idx ,psnr, ssim))

                        avg_psnr += psnr
                        avg_ssim += ssim

                        logs = diffusion.get_current_log()
                        if wandb_logger:
                            wandb_logger.log_metrics(logs)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}, # SSIM: {:.4e}'.format(avg_psnr, avg_ssim))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:5d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr, avg_ssim))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_ssim': avg_ssim,
                            'val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0 and current_step >= opt['train']['over_val']:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        logger_val = logging.getLogger('val')  # validation logger
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_loader in enumerate(val_loaders):
            val_i = 0
            psnr = 0.0
            ssim = 0.0
            val_psnr = 0.0
            val_ssim = 0.0
            sr_imgs = []
            hr_imgs = []
            fake_imgs = []
            idx += 1
            for _, val_data in enumerate(val_loader):
                diffusion.feed_data(val_data)
                # if torch.numel(val_data['HR'][0][0]) != torch.numel(val_data['HR'][0][0])-torch.count_nonzero(val_data['HR'][0][0]).item():
                diffusion.test(continous=False)
                visuals = diffusion.get_current_visuals()
                for i in range(visuals['SR'].shape[0]):
                    hr_patch = Metrics.tensor2mhd(visuals['HR'][i])
                    sr_patch = Metrics.tensor2mhd(visuals['SR'][i])
                    fake_patch = Metrics.tensor2mhd(visuals['INF'][i])
                    if val_i == opt['train']['val_i']:
                        val_img = np.concatenate([fake_patch, sr_patch, hr_patch], axis=1) # uint8
                        Metrics.save_mhd(val_img, '{}/{}_{}_val.mhd'.format(result_path, current_step, idx))
                        val_psnr = Metrics.calculate_psnr(sr_patch, hr_patch)
                        val_ssim = Metrics.calculate_ssim(sr_patch, hr_patch)
                    sr_imgs.append(sr_patch)  # uint8
                    hr_imgs.append(hr_patch)  # uint8
                    fake_imgs.append(fake_patch)  # uint8
                    val_i += 1
            #patchの再構成
            sr_img = Metrics.concatImage(sr_imgs, opt)
            hr_img = Metrics.concatImage(hr_imgs, opt)
            fake_img = Metrics.concatImage(fake_imgs, opt)
            # save
            Metrics.save_mhd(hr_img, '{}/{}_{}_hr.mhd'.format(result_path, current_step, idx))
            Metrics.save_mhd(sr_img, '{}/{}_{}_sr.mhd'.format(result_path, current_step, idx))
            Metrics.save_mhd(fake_img, '{}/{}_{}_inf.mhd'.format(result_path, current_step, idx))

            # generation
            mask = mask_imgs[idx-1]
            psnr = Metrics.calculate_psnr_mask(sr_img, hr_img, mask)
            ssim = Metrics.calculate_ssim_mask(sr_img, hr_img, mask)
            logger.info('# Validation_patch{} # PSNR: {:.4e}, # SSIM: {:.4e}'.format(idx ,val_psnr, val_ssim))
            logger_val.info('# Validation{} # PSNR: {:.4e}, # SSIM: {:.4e}'.format(idx ,psnr, ssim))

            avg_psnr += psnr
            avg_ssim += ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, sr_img, hr_img, psnr, ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
