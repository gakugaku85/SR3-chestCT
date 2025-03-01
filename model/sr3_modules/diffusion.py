import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gudhi.wasserstein import wasserstein_distance
# from torch_topological.nn import CubicalComplex
# from torch_topological.nn import WassersteinDistance
import gudhi as gd
import multiprocessing
from skimage.filters import frangi
from icecream import ic
import time

from model.torch_topological.nn import CubicalComplex
from model.torch_topological.nn import WassersteinDistance


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class TopologicalWDLoss(nn.Module):
    def __init__(self, weight_thre=0.01):
        super(TopologicalWDLoss, self).__init__()
        self.cubical = CubicalComplex()
        self.wd_loss_func = WassersteinDistance(q=2)
        self.weight_thre = weight_thre

    def forward(self, hr_batch, sr_batch, frangi_batch):

        loss = 0.0
        start_time = time.time()
        cubical_time = []
        frangi_time = []
        select_time = []
        wd_time = []

        for hr, sr, frangi_img in zip(hr_batch, sr_batch, frangi_batch):
            cubical_start = time.time()
            hr = hr.squeeze()
            sr = sr.squeeze()
            frangi_img = frangi_img.squeeze()

            hr_info = self.cubical(1.0-hr)[0] # 0はpersistence diagramの連結成分、1は穴
            sr_info = self.cubical(1.0-sr)[0]
            cubical_time.append(time.time() - cubical_start)

            # frangi_start = time.time()
            # frangi_img = frangi(1.0 - hr.cpu().numpy())
            # frangi_img = torch.tensor(frangi_img, device=hr.device)
            # frangi_time.append(time.time() - frangi_start)

            def select_point_by_weight(diagram):
                birth_death = diagram.diagram.clone().detach().to(hr.device)  # shape: (N, 2)
                pair_indices = diagram.pairing.clone().detach().to(hr.device)[:, :2]  # shape: (N, 2)

                # birth, death を分離
                birth = birth_death[:, 0]
                death = birth_death[:, 1]

                distance = torch.abs(birth - death) / torch.sqrt(torch.tensor(2.0, device=hr.device))

                weights = distance * frangi_img[pair_indices[:, 0], pair_indices[:, 1]]

                valid_indices = weights > self.weight_thre
                valid_birth_death = birth_death[valid_indices]
                return valid_birth_death.unsqueeze(0)

            select_start = time.time()
            per_hr = select_point_by_weight(hr_info)
            select_time.append(time.time() - select_start)

            # ic(len(per_hr), len(per_hr[0]))
            # ic(len(sr_info.diagram), len(sr_info.diagram[0]))
            wd_start = time.time()
            loss += self.wd_loss_func(per_hr, sr_info)
            wd_time.append(time.time() - wd_start)

        # ic("cubical", np.sum(cubical_time))
        # ic("select", np.sum(select_time))
        # ic("wd", np.sum(wd_time))
        # ic("total", time.time() - start_time)
        return loss

def match_cofaces_with_gudhi(image_data, cofaces):
    height, width = image_data.shape
    result = []

    for dim, pairs in enumerate(cofaces[0]):
        for birth, death in pairs:
            birth_y, birth_x = np.unravel_index(birth, (height, width))
            death_y, death_x = np.unravel_index(death, (height, width))
            pers = (1.00-image_data.ravel()[birth], 1.00-image_data.ravel()[death])
            result.append((dim, pers,((birth_y, birth_x), (death_y, death_x))))

    for dim, births in enumerate(cofaces[1]):
        for birth in births:
            birth_y, birth_x = np.unravel_index(birth, (height, width))
            pers = (1.00-image_data.ravel()[birth], 1.0)
            result.append((dim, pers, ((birth_y, birth_x), None)))

    return result

def persistent_homology(image_data, weight_threshold=0.01, hr_or_sr="hr"):
    """Computes and visualizes the persistent homology for the given image data."""
    cc = gd.CubicalComplex(
        dimensions=image_data.shape, top_dimensional_cells=1 - image_data.flatten()
    )
    cc.persistence()
    cofaces = cc.cofaces_of_persistence_pairs()
    result = match_cofaces_with_gudhi(image_data=image_data, cofaces=cofaces)

    frangi_img = frangi(1-image_data)
    new_result = []

    for dim, (birth, death) , coordinates in result:
        if dim == 1:
            continue
        distance = np.abs(birth - death) / np.sqrt(2)
        weight = distance * frangi_img[coordinates[0][0], coordinates[0][1]]

        if weight > weight_threshold:
            new_result.append([birth, death])

    return np.array(new_result)

class WassersteinDistanceLoss(nn.Module):
    def __init__(self):
        super(WassersteinDistanceLoss, self).__init__()

    def process_image(self, hr_img, sr_img, index, losses):

        hr_connect = persistent_homology(hr_img[0], hr_or_sr="hr")
        sr_connect = persistent_homology(sr_img[0], hr_or_sr="sr")
        losses[index] = wasserstein_distance(hr_connect, sr_connect)

    def forward(self, hr, sr):
        num_images = hr.shape[0]

        # multiprocessing.Managerを使って共有リストを作成
        with multiprocessing.Manager() as manager:
            # 共有リストを作成
            losses = manager.list([None] * num_images)
            processes = []

            # 各プロセスを開始
            for i in range(num_images):
                hr_img = hr[i].detach().float().cpu().numpy()
                sr_img = sr[i].detach().float().cpu().numpy()
                process = multiprocessing.Process(target=self.process_image, args=(
                    hr_img, sr_img, i, losses))
                processes.append(process)
                process.start()

            # 全プロセスの終了を待つ
            for process in processes:
                process.join()

            losses = [l for l in losses if l is not None]

            return torch.tensor(sum(losses) / len(losses)) if len(losses) > 0 else torch.tensor(0.0)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        under_step_wd_loss=2000,
        conditional=True,
        schedule_opt=None,
        weight_thre=0.01,
        wdloss_weight=100
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.under_step_wd_loss = under_step_wd_loss
        self.weight_thre = weight_thre
        self.wdloss_weight = wdloss_weight
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        elif self.loss_type == 'wd':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
            self.loss_func2 = TopologicalWDLoss(weight_thre=self.weight_thre).to(device)
        else:
            raise NotImplementedError()
        # Sobelフィルタのカーネルを定義

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        gamma = np.cumprod(alphas, axis=0)
        gamma_prev = np.append(1., gamma[:-1])
        self.sqrt_gamma_prev = np.sqrt(np.append(1., gamma))
        # print(gamma)
        # print(gamma_prev)
        # print(self.sqrt_gamma_prev)
        # input()
        plt.plot(gamma)
        plt.ylabel('gamma')
        plt.xlabel('timesteps')
        plt.savefig('./gamma.png')

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('gamma', to_torch(gamma))
        self.register_buffer('gamma_prev', to_torch(gamma_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(gamma)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - gamma)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - gamma)))
        self.register_buffer('sqrt_recip_gamma',
                             to_torch(np.sqrt(1. / gamma)))
        self.register_buffer('sqrt_recipm1_gamma',
                             to_torch(np.sqrt(1. / gamma - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gamma_prev) / (1. - gamma)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(gamma_prev) / (1. - gamma)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - gamma_prev) * np.sqrt(alphas) / (1. - gamma)))

    def predict_start_from_noise(self, x_t, t, noise): #predict y0
        return self.sqrt_recip_gamma[t] * x_t - \
            self.sqrt_recipm1_gamma[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_gamma_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return img #バッチサイズが欠落しないように変更した
            # return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']

        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_gamma_prev[t-1],
                self.sqrt_gamma_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        self.origin_loss = self.loss_func(noise, x_recon)
        if self.loss_type == 'wd':
            self.wd_loss = torch.tensor(0.0).to(x_start.device)
            if t < self.under_step_wd_loss:
                denoise_img = self.predict_start_from_noise(x_noisy, t=t-1, noise=x_recon)
                self.wd_loss = self.wdloss_weight * self.loss_func2(x_in['HR'], denoise_img, x_in['Frangi'])
                # self.wd_loss = self.loss_func2(x_in['HR'], denoise_img)
            loss = self.origin_loss + self.wd_loss
            # print("wd loss: ", self.wd_loss)
            # print("origin loss: ", self.origin_loss)
            # print("total loss: ", loss)
        elif self.loss_type == 'l1':
            loss = self.origin_loss
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

    def print_train_result(self):
        return self.train_result

    def get_each_loss(self):
        if self.loss_type == 'wd':
            return self.origin_loss, self.wd_loss
        elif self.loss_type == 'l1':
            return self.origin_loss, torch.tensor(0.0).to(self.origin_loss.device)