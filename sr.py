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
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="2")
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

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
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'val':
            val_sets = Data.create_dataset(dataset_opt, phase)
            val_loaders = [Data.create_dataloader(val_set, dataset_opt, phase) for val_set in val_sets]
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in tqdm(enumerate(train_loader), desc='training step', total=len(train_loader)):
                # result_path = '{}/{}'.format(opt['path']['results'], current_step)
                # os.makedirs(result_path+"iter", exist_ok=True)

                current_step += 1
                # if current_step > n_iter:
                #     break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    # visuals = diffusion.get_current_visuals()
                    # sr_img = Metrics.tensor2mhd(visuals['SR'])  # uint8

                    # Metrics.save_mhd(sr_img, '{}iter/{}_sr.mhd'.format(result_path, current_step))
                    # if wandb_logger:
                    #     wandb_logger.log_image(
                    #         f'iter_{current_step}', 
                    #         sr_img
                    #     )

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')

                    for _,  val_loader in enumerate(val_loaders):
                        sr_imgs = []
                        hr_imgs = []
                        lr_imgs = []
                        fake_imgs = []
                        for _, val_data in tqdm(enumerate(val_loader), desc='sampling loop time step', total=len(val_loader)):
                            diffusion.feed_data(val_data)
                            diffusion.test(continous=False)
                            visuals = diffusion.get_current_visuals()
                            sr_imgs.append(Metrics.tensor2mhd(visuals['SR']))  # uint8
                            hr_imgs.append(Metrics.tensor2mhd(visuals['HR']))  # uint8
                            lr_imgs.append(Metrics.tensor2mhd(visuals['LR']))  # uint8
                            fake_imgs.append(Metrics.tensor2mhd(visuals['INF']))  # uint8
                        sr_img = Metrics.concatImage(sr_imgs, opt['datasets']['val']['image_h'], opt['datasets']['val']['image_w'])
                        hr_img = Metrics.concatImage(hr_imgs, opt['datasets']['val']['image_h'], opt['datasets']['val']['image_w'])
                        lr_img = Metrics.concatImage(lr_imgs, opt['datasets']['val']['image_h'], opt['datasets']['val']['image_w'])
                        fake_img = Metrics.concatImage(fake_imgs, opt['datasets']['val']['image_h'], opt['datasets']['val']['image_w'])
                        idx += 1
                            # generation
                        Metrics.save_mhd(
                            hr_img, '{}/{}_{}_hr.mhd'.format(result_path, current_step, idx))
                        Metrics.save_mhd(
                            sr_img, '{}/{}_{}_sr.mhd'.format(result_path, current_step, idx))
                        Metrics.save_mhd(
                            lr_img, '{}/{}_{}_lr.mhd'.format(result_path, current_step, idx))
                        Metrics.save_mhd(
                            fake_img, '{}/{}_{}_inf.mhd'.format(result_path, current_step, idx))
                        # tb_logger.add_image(
                        #     'Iter_{}'.format(current_step),
                        #     np.transpose(np.concatenate(
                        #         (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                        #     idx)
                        avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)

                        logs = diffusion.get_current_log()
                        if wandb_logger:
                            wandb_logger.log_metrics(logs)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    # avg_psnr = avg_psnr / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics(avg_psnr)
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    # else:
    #     logger.info('Begin Model Evaluation.')
    #     avg_psnr = 0.0
    #     avg_ssim = 0.0
    #     idx = 0
    #     result_path = '{}'.format(opt['path']['results'])
    #     os.makedirs(result_path, exist_ok=True)
    #     for _,  val_data in enumerate(val_loader):
    #         idx += 1
    #         diffusion.feed_data(val_data)
    #         diffusion.test(continous=True)
    #         visuals = diffusion.get_current_visuals()

    #         hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
    #         lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
    #         fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
    #         sr_map_img = Metrics.tensor2img(visuals['SR'][-1])
    #         map_img = (hr_img - sr_map_img)**2

    #         sr_img_mode = 'grid'
    #         if sr_img_mode == 'single':
    #             # single img series
    #             sr_img = visuals['SR']  # uint8
    #             sample_num = sr_img.shape[0]
    #             for iter in range(0, sample_num):
    #                 Metrics.save_img(
    #                     Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
    #         else:
    #             # grid img
    #             sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
    #             Metrics.save_img(
    #                 sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
    #             Metrics.save_img(
    #                 Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

    #         Metrics.save_img(
    #             hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
    #         Metrics.save_img(
    #             lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
    #         Metrics.save_img(
    #             fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
    #         Metrics.save_img(
    #             map_img, '{}/{}_{}_2map.png'.format(result_path, current_step, idx))

    #         # generation
    #         eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
    #         eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

    #         avg_psnr += eval_psnr
    #         avg_ssim += eval_ssim

    #         if wandb_logger and opt['log_eval']:
    #             wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

    #     avg_psnr = avg_psnr / idx
    #     avg_ssim = avg_ssim / idx

    #     # log
    #     logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    #     logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
    #     logger_val = logging.getLogger('val')  # validation logger
    #     logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
    #         current_epoch, current_step, avg_psnr, avg_ssim))

    #     if wandb_logger:
    #         if opt['log_eval']:
    #             wandb_logger.log_eval_table()
    #         wandb_logger.log_metrics({
    #             'PSNR': float(avg_psnr),
    #             'SSIM': float(avg_ssim)
    #         })
