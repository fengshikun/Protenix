# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional

import torch

from protenix.model.utils import centre_random_augmentation
import numpy as np

class PreCond:
    def __init__(self, ns):
        raise NotImplementedError

    def _get_scalings_and_weightings(self, t):
        raise NotImplementedError

    def get_scalings_and_weightings(self, t, ndim):
        c_skip, c_in, c_out, c_noise, weightings = self._get_scalings_and_weightings(t)
        c_skip, c_in, c_out, weightings = [append_dims(item, ndim) for item in [c_skip, c_in, c_out, weightings]]
        return c_skip, c_in, c_out, c_noise, weightings


class DDBMPreCond(PreCond):
    def __init__(self, ns, sigma_data, cov_xy):
        self.ns, self.sigma_data, self.cov_xy = ns, sigma_data, cov_xy
        self.sigma_data_end = sigma_data

    def _get_scalings_and_weightings(self, t):
        a_t, b_t, c_t = self.ns.get_abc(t)
        A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2 * a_t * b_t * self.cov_xy + c_t**2
        c_in = 1 / (A) ** 0.5
        c_skip = (b_t * self.sigma_data**2 + a_t * self.cov_xy) / A
        c_out = (
            a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * c_t**2
        ) ** 0.5 * c_in
        c_noise = 1000 * 0.25 * torch.log(t + 1e-44)
        weightings = 1 / c_out**2
        return c_skip, c_in, c_out, c_noise, weightings

class TrainingNoiseSampler:
    """
    Sample the noise-level of of training samples
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Sampler for training noise-level

        Args:
            p_mean (float, optional): gaussian mean. Defaults to -1.2.
            p_std (float, optional): gaussian std. Defaults to 1.5.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        print(f"train scheduler {self.sigma_data}")

    def __call__(
        self, size: torch.Size, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sampling

        Args:
            size (torch.Size): the target size
            device (torch.device, optional): target device. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: sampled noise-level
        """
        rnd_normal = torch.randn(size=size, device=device)
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        return noise_level


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps)
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Scheduler parameters

        Args:
            s_max (float, optional): maximal noise level. Defaults to 160.0.
            s_min (float, optional): minimal noise level. Defaults to 4e-4.
            rho (float, optional): the exponent numerical part. Defaults to 7.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        print(f"inference scheduler {self.sigma_data}")

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list

class RealUniformSampler:
    def __init__(self, sigma_max=80, sigma_min=0.002):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def sample(self, batch_size, device):
        ts = torch.rand(batch_size).to(device) *(self.sigma_max - self.sigma_min) + self.sigma_min
        return ts, torch.ones_like(ts)
        # ts = torch.rand(batch_size).to(device) *(self.sigma_max**2 - self.sigma_min**2) + self.sigma_min**2
        # return torch.sqrt(ts), torch.ones_like(ts)

class RealUniformSamplerSquare:
    def __init__(self, sigma_max=80, sigma_min=0.002):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def sample(self, batch_size, device):
        ts = torch.rand(batch_size).to(device) *(self.sigma_max**2 - self.sigma_min**2) + self.sigma_min**2
        return ts, torch.ones_like(ts)

class RealUnifromSamplerLogisticnorm: 
    def __init__(self, sigma_max=80, sigma_min=0.002, lognorm_mean=0.0, lognorm_std=1.0): 
        self.sigma_max = sigma_max 
        self.sigma_min = sigma_min 
        self.lognorm_mean = lognorm_mean
        self.lognorm_std = lognorm_std
    
    def sample(self, batch_size, device): # LogNormal sampling 
        # ts = torch.distributions.LogisticNormal( self.lognorm_mean, self.lognorm_std ).sample((batch_size,)).to(device) # 若你仍然希望限制在 [sigma_min, sigma_max] 
        z = torch.randn(batch_size, device=device) * self.lognorm_std + self.lognorm_mean
        u = torch.sigmoid(z)  # (0,1)
        
        ts = u*(self.sigma_max**2 - self.sigma_min**2) + self.sigma_min**2 
        return torch.sqrt(ts), torch.ones_like(ts)

# class RealUnifromSamplerLogisticnorm:
#     def __init__(
#         self,
#         sigma_max=80,
#         sigma_min=0.002,
#         lognorm_mean=0.0,
#         lognorm_std=1.0,
#     ):
#         self.sigma_max = sigma_max
#         self.sigma_min = sigma_min
#         self.lognorm_mean = lognorm_mean
#         self.lognorm_std = lognorm_std

#     def sample(self, batch_size, device):
#         # LogNormal sampling
#         ts = torch.distributions.LogNormal(
#             self.lognorm_mean,
#             self.lognorm_std
#         ).sample((batch_size,)).to(device)

#         # 若你仍然希望限制在 [sigma_min, sigma_max]
#         ts = ts*(self.sigma_max - self.sigma_min) + self.sigma_min

#         return ts, torch.ones_like(ts)


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

class KarrasSigmaSampler:
    """
    Noise schedule from Karras et al. (2022)
    """
    def __init__(self, sigma_min, sigma_max, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        # 对应原来的 n
        ramp = torch.linspace(0, 1, N_step, device=device, dtype=dtype)

        min_inv_rho = self.sigma_min ** (1.0 / self.rho)
        max_inv_rho = self.sigma_max ** (1.0 / self.rho)

        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        sigmas = append_zero(sigmas)

        # 与 RealUniformSampler 对齐，返回 (values, weights)
        return sigmas



class UniformSigmaSampler:
    """
    Uniform noise schedule sampler.
    
    Generates linearly spaced sigmas from t_max to t_min.
    Returns (sigmas, weights) similar to other samplers.
    """
    def __init__(self, t_min: float = 0.0001, t_max: float = 1.0 - 1e-3):
        self.t_min = t_min
        self.t_max = t_max

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Args:
            N_step: number of steps (like 'n' in get_sigmas_uniform)
            device: torch device
            dtype: torch dtype

        Returns:
            sigmas: Tensor of shape (N_step+1,), last value appended 0
        """
        sigmas = torch.linspace(self.t_max, self.t_min, N_step + 1, device=device, dtype=dtype)
        return sigmas




class NoiseSchedule:
    def __init__(self):
        raise NotImplementedError

    def get_f_g2(self, t):
        raise NotImplementedError

    def get_alpha_rho(self, t):
        raise NotImplementedError

    def get_abc(self, t):
        alpha_t, alpha_bar_t, rho_t, rho_bar_t = self.get_alpha_rho(t)
        a_t, b_t, c_t = (
            (alpha_bar_t * rho_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t * rho_t) / self.rho_T,
        )
        a_t = a_t.to(t.dtype)
        b_t = b_t.to(t.dtype)
        c_t = c_t.to(t.dtype)
        return a_t, b_t, c_t


class VPNoiseSchedule(NoiseSchedule):
    def __init__(self, beta_d=2, beta_min=0.1):
        self.beta_d, self.beta_min = beta_d, beta_min
        self.alpha_fn = lambda t: np.e ** (-0.5 * beta_min * t - 0.25 * beta_d * t**2)
        self.alpha_T = self.alpha_fn(1)
        self.rho_fn = lambda t: (np.e ** (beta_min * t + 0.5 * beta_d * t**2) - 1).sqrt()
        self.rho_T = self.rho_fn(torch.DoubleTensor([1])).item()

        self.f_fn = lambda t: (-0.5 * beta_min - 0.5 * beta_d * t)
        self.g2_fn = lambda t: (beta_min + beta_d * t)

    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        
        alpha_t = alpha_t.to(torch.float32)
        alpha_bar_t = alpha_bar_t.to(torch.float32)
        rho_t = rho_t.to(torch.float32)
        rho_bar_t = rho_bar_t.to(torch.float32)
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


class VENoiseSchedule(NoiseSchedule):
    def __init__(self, sigma_max=80.0):
        self.sigma_max = sigma_max
        self.alpha_fn = lambda t: torch.ones_like(t)
        self.alpha_T = 1
        self.rho_fn = lambda t: t
        self.rho_T = sigma_max

        self.f_fn = lambda t: torch.zeros_like(t)
        self.g2_fn = lambda t: 2 * t

    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        
        
        alpha_t = alpha_t.to(torch.float32)
        alpha_bar_t = alpha_bar_t.to(torch.float32)
        rho_t = rho_t.to(torch.float32)
        rho_bar_t = rho_bar_t.to(torch.float32)
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


class BatchedSeedGenerator:

    def __init__(self, seeds=None):
        self.num_samples = len(seeds)
        if torch.cuda.is_available():
            self.rng = [torch.Generator(torch.device('cuda')) for _ in range(self.num_samples)]
        else:
            self.rng = [torch.Generator() for _ in range(self.num_samples)]
        [rng.manual_seed(int(seeds[i])) for i, rng in enumerate(self.rng)]

    def randn(self, size, dtype=torch.float, device="cpu"):
        assert size[0] == self.num_samples
        return torch.cat(
            [
                torch.randn(1, *size[1:], generator=self.rng[i], dtype=dtype, device=device)
                for i in range(self.num_samples)
            ],
            dim=0,
        )

    def randint(self, low, high, size, dtype=torch.long, device="cpu"):
        assert size[0] == self.num_samples
        return torch.cat(
            [
                torch.randint(
                    low,
                    high,
                    generator=self.rng[i],
                    size=(1, *size[1:]),
                    dtype=dtype,
                    device=device,
                )
                for i in range(self.num_samples)
            ],
            dim=0,
        )

    def randn_like(self, tensor):
        size, dtype, device = tensor.size(), tensor.dtype, tensor.device
        return self.randn(size, dtype=dtype, device=device)


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        # init noise
        # [..., N_sample, N_atom, 3]
        x_l = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )  # NOTE: set seed in distributed training

        for _, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            # [..., N_sample, N_atom, 3]
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )

            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            # Euler step only
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            x_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
            )

            delta = (x_noisy - x_denoised) / t_hat[
                ..., None, None
            ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

        return x_l

    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        x_l = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            chunk_x_l = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe
            )
            x_l.append(chunk_x_l)
        x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]
    return x_l

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def to_d(x, sigma, denoised, x_T, sigma_max,   w=1, stochastic=False):
    """Converts a denoiser output to a Karras ODE derivative."""
    grad_pxtlx0 = (denoised - x) / append_dims(sigma**2, x.ndim)
    grad_pxTlxt = (x_T - x) / (append_dims(torch.ones_like(sigma)*sigma_max**2, x.ndim) - append_dims(sigma**2, x.ndim))
    gt2 = 2*sigma
    d = - (0.5 if not stochastic else 1) * gt2 * (grad_pxtlx0 - w * grad_pxTlxt * (0 if stochastic else 1))
    if stochastic:
        return d, gt2
    else:
        return d


def get_d_vp(x, denoised, x_T, std_t,logsnr_t, logsnr_T, logs_t, logs_T, s_t_deriv, sigma_t, sigma_t_deriv, w, stochastic=False):
    
    a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
    b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
    
    mu_t = a_t * x_T + b_t * denoised 
    
    grad_logq = - (x - mu_t)/std_t**2 / (-torch.expm1(logsnr_T - logsnr_t))
    # grad_logpxtlx0 = - (x - logs_t.exp()*denoised)/std_t**2 
    grad_logpxTlxt = -(x - torch.exp(logs_t-logs_T)*x_T) /std_t**2  / torch.expm1(logsnr_t - logsnr_T)

    f = s_t_deriv * (-logs_t).exp() * x
    gt2 = 2 * (logs_t).exp()**2 * sigma_t * sigma_t_deriv 
    # breakpoint()

    d = f -  gt2 * ((0.5 if not stochastic else 1)* grad_logq - w * grad_logpxTlxt)
    # d = f - (0.5 if not stochastic else 1) * gt2 * (grad_logpxtlx0 - w * grad_logpxTlxt* (0 if stochastic else 1))
    if stochastic:
        return d, gt2
    else:
        return d

def bridge_sample_dbim(x0, xT, t, noise, noise_schedule):
    a_t, b_t, c_t = [append_dims(item, x0.ndim) for item in noise_schedule.get_abc(t)]
    samples = a_t * xT + b_t * x0 + c_t * noise
    return samples

def sample_diffusion_ddbm(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    ddbm_configs: dict[str, Any] = None,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    
    
    
    def sample_dbim(
        denoiser,
        x,
        ts,
        noise_schedule,
        t_max,
        eta=1.0,
        mask=None,
        seed=42,
        **kwargs,
    ):
        x_T = x
        path = []
        pred_x0 = []

        ones = x.new_ones([x.shape[0]])
        indices = range(len(ts) - 1)

        nfe = 0
        # x0_hat = denoiser(x, t_max * ones)
        
        batch_sigmas = t_max * ones
        precond = DDBMPreCond(noise_schedule, ddbm_configs["sigma_data"], ddbm_configs["cov_xy"])
        c_skip, c_in, c_out, c_noise , _ = precond.get_scalings_and_weightings(batch_sigmas, x.ndim)
        # convert c_skip, c_in, c_out to dtype of x
        # c_skip = c_skip.to(x.dtype)
        # c_in = c_in.to(x.dtype)
        # c_out = c_out.to(x.dtype)
        x0_hat = denoiser(
                    x_noisy=x,
                    t_hat_noise_level=batch_sigmas, # not matter when provide the c_in, c_skip and c_out
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    c_in=c_in,
                    c_skip=c_skip,
                    c_out=c_out,
                )
                
                # denoised = denoiser(x, sigmas[i] * s_in, x_T)
                # denoised = denoiser(x, sigmas[i] * s_in, x_T)
        
        # repeat seed for each sample in the batch
        batch_num = x.shape[0]
        seed = [seed for _ in range(batch_num)]
        generator = BatchedSeedGenerator(seed)
        noise = generator.randn_like(x0_hat)
        first_noise = noise
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)
        x = bridge_sample_dbim(x0_hat, x_T, ts[0] * ones, noise, noise_schedule)
        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())
        nfe += 1

        for _, i in enumerate(indices):
            s = ts[i]
            t = ts[i + 1]

            batch_sigmas = s * ones
            c_skip, c_in, c_out, c_noise , _ = precond.get_scalings_and_weightings(batch_sigmas, x.ndim)
            # convert c_skip, c_in, c_out to dtype of x
            # c_skip = c_skip.to(x.dtype)
            # c_in = c_in.to(x.dtype)
            # c_out = c_out.to(x.dtype)
            x0_hat = denoiser(
                        x_noisy=x,
                        t_hat_noise_level=batch_sigmas, # not matter when provide the c_in, c_skip and c_out
                        input_feature_dict=input_feature_dict,
                        s_inputs=s_inputs,
                        s_trunk=s_trunk,
                        z_trunk=z_trunk,
                        chunk_size=attn_chunk_size,
                        inplace_safe=inplace_safe,
                        c_in=c_in,
                        c_skip=c_skip,
                        c_out=c_out,
                    )
            # x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)

            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in noise_schedule.get_abc(t * ones)]

            _, _, rho_s, _ = [append_dims(item, x0_hat.ndim) for item in noise_schedule.get_alpha_rho(s * ones)]
            alpha_t, _, rho_t, _ = [
                append_dims(item, x0_hat.ndim) for item in noise_schedule.get_alpha_rho(t * ones)
            ]

            omega_st = eta * (alpha_t * rho_t) * (1 - rho_t**2 / rho_s**2).sqrt()
            tmp_var = (c_t**2 - omega_st**2).sqrt() / c_s
            coeff_xs = tmp_var
            coeff_x0_hat = b_t - tmp_var * b_s
            coeff_xT = a_t - tmp_var * a_s

            noise = generator.randn_like(x0_hat)

            x = coeff_x0_hat * x0_hat + coeff_xT * x_T + coeff_xs * x + (1 if i != len(ts) - 2 else 0) * omega_st * noise

            path.append(x.detach().cpu())
            pred_x0.append(x0_hat.detach().cpu())
            nfe += 1

        return x, path, nfe, pred_x0, ts, first_noise


    def sample_heun(
        denoiser,
        x,
        sigmas,
        pred_mode='both',
        progress=False,
        callback=None,
        sigma_max=1.0,
        beta_d=2,
        beta_min=0.1,
        churn_step_ratio=0.33,
        guidance=1,
    ):
        """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
        x_T = x
        path = [x]
        
        s_in = x.new_ones([x.shape[0]])
        indices = range(len(sigmas) - 1)
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        nfe = 0
        assert churn_step_ratio < 1

        if pred_mode.startswith('vp'):
            vp_snr_sqrt_reciprocal = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
            vp_snr_sqrt_reciprocal_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (vp_snr_sqrt_reciprocal(t) + 1 / vp_snr_sqrt_reciprocal(t))
            s = lambda t: (1 + vp_snr_sqrt_reciprocal(t) ** 2).rsqrt()
            s_deriv = lambda t: -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)

            logs = lambda t: -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min
            
            std =  lambda t: vp_snr_sqrt_reciprocal(t) * s(t)
            
            logsnr = lambda t :  - 2 * torch.log(vp_snr_sqrt_reciprocal(t))

            logsnr_T = logsnr(torch.as_tensor(sigma_max))
            logs_T = logs(torch.as_tensor(sigma_max))
        
        for j, i in enumerate(indices):
            
            if churn_step_ratio > 0:
                # 1 step euler
                sigma_hat = (sigmas[i+1] - sigmas[i]) * churn_step_ratio + sigmas[i]
                
                batch_sigmas = sigmas[i] * s_in
                c_skip, c_out, c_in = get_bridge_scalings(ddbm_configs, batch_sigmas)
    
                # expand dim
                c_skip = c_skip.view(c_skip.shape + (1,) * (x.dim() - c_skip.dim()))
                c_out = c_out.view(c_out.shape + (1,) * (x.dim() - c_out.dim()))
                c_in = c_in.view(c_in.shape + (1,) * (x.dim() - c_in.dim()))
                denoised = denoiser(
                    x_noisy=x,
                    t_hat_noise_level=batch_sigmas, # not matter when provide the c_in, c_skip and c_out
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    c_in=c_in,
                    c_skip=c_skip,
                    c_out=c_out,
                )
                
                # denoised = denoiser(x, sigmas[i] * s_in, x_T)
                # denoised = denoiser(x, sigmas[i] * s_in, x_T)
                if pred_mode == 've':
                    d_1, gt2 = to_d(x, sigmas[i] , denoised, x_T, sigma_max,  w=guidance, stochastic=True)
                elif pred_mode.startswith('vp'):
                    d_1, gt2 = get_d_vp(x, denoised, x_T, std(sigmas[i]),logsnr(sigmas[i]), logsnr_T, logs(sigmas[i] ), logs_T, s_deriv(sigmas[i] ), vp_snr_sqrt_reciprocal(sigmas[i] ), vp_snr_sqrt_reciprocal_deriv(sigmas[i] ), guidance, stochastic=True)
                
                dt = (sigma_hat - sigmas[i]) 
                x = x + d_1 * dt + torch.randn_like(x) *((dt).abs() ** 0.5)*gt2.sqrt()
                
                nfe += 1
                
                path.append(x.detach().cpu())
            else:
                sigma_hat =  sigmas[i]
            
            # heun step
            # denoised = denoiser(x, sigmas[i] * s_in, x_T)
            denoised = denoiser(
                    x_noisy=x,
                    t_hat_noise_level=batch_sigmas, # not matter when provide the c_in, c_skip and c_out
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    c_in=c_in,
                    c_skip=c_skip,
                    c_out=c_out,
            )
            # denoised = denoiser(x, sigma_hat * s_in, x_T)
            if pred_mode == 've':
                # d =  (x - denoised ) / append_dims(sigma_hat, x.ndim)
                d = to_d(x, sigma_hat, denoised, x_T, sigma_max, w=guidance)
            elif pred_mode.startswith('vp'):
                d = get_d_vp(x, denoised, x_T, std(sigma_hat),logsnr(sigma_hat), logsnr_T, logs(sigma_hat), logs_T, s_deriv(sigma_hat), vp_snr_sqrt_reciprocal(sigma_hat), vp_snr_sqrt_reciprocal_deriv(sigma_hat), guidance)
                
            nfe += 1
            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": i,
                        "sigma": sigmas[i],
                        "sigma_hat": sigma_hat,
                        "denoised": denoised,
                    }
                )
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                
                x = x + d * dt 
                
            else:
                # Heun's method
                x_2 = x + d * dt    
                batch_sigmas2 = sigmas[i+1] * s_in
                c_skip2, c_out2, c_in2 = get_bridge_scalings(ddbm_configs, batch_sigmas2)
    
                # expand dim
                c_skip2 = c_skip2.view(c_skip.shape + (1,) * (x.dim() - c_skip.dim()))
                c_out2 = c_out2.view(c_out.shape + (1,) * (x.dim() - c_out.dim()))
                c_in2 = c_in2.view(c_in.shape + (1,) * (x.dim() - c_in.dim()))
                denoised_2 = denoiser(
                    x_noisy=x_2,
                    t_hat_noise_level=batch_sigmas2, # not matter when provide the c_in, c_skip and c_out
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    c_in=c_in2,
                    c_skip=c_skip2,
                    c_out=c_out2,
                )
                
                # denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, x_T)
                if pred_mode == 've':
                    # d_2 =  (x_2 - denoised_2) / append_dims(sigmas[i + 1], x.ndim)
                    d_2 = to_d(x_2,  sigmas[i + 1], denoised_2, x_T, sigma_max, w=guidance)
                elif pred_mode.startswith('vp'):
                    d_2 = get_d_vp(x_2, denoised_2, x_T, std(sigmas[i + 1]),logsnr(sigmas[i + 1]), logsnr_T, logs(sigmas[i + 1]), logs_T, s_deriv(sigmas[i + 1]),
                                    vp_snr_sqrt_reciprocal(sigmas[i + 1]), vp_snr_sqrt_reciprocal_deriv(sigmas[i + 1]), guidance)
                
                d_prime = (d + d_2) / 2

                # noise = torch.zeros_like(x) if 'flow' in pred_mode or pred_mode == 'uncond' else generator.randn_like(x)
                x = x + d_prime * dt #+ noise * (sigmas[i + 1]**2 - sigma_hat**2).abs() ** 0.5
                nfe += 1
            # loss = (denoised.detach().cpu() - x0).pow(2).mean().item()
            # losses.append(loss)

            path.append(x.detach().cpu())
            
        return x, path, nfe



    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        # init noise
        # [..., N_sample, N_atom, 3]
        x_l = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )  # NOTE: set seed in distributed training

        for _, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            # [..., N_sample, N_atom, 3]
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )

            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            # Euler step only
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            x_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
            )

            delta = (x_noisy - x_denoised) / t_hat[
                ..., None, None
            ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

        return x_l

    # repeat the apo coordinates N_sample times
    x_T = input_feature_dict['apo_atom_array'].unsqueeze(0).repeat(N_sample, 1, 1)
    
    test_sampler = ddbm_configs.get("infer_sampler", "ddbm")
    if test_sampler == 'dbim':
        if ddbm_configs.get("pred_mode", "vp") == "ve":
            noise_schedule_obj = VENoiseSchedule(sigma_max=ddbm_configs["sigma_max"])
        elif ddbm_configs.get("pred_mode", "vp").startswith("vp"):
            noise_schedule_obj = VPNoiseSchedule(beta_d=ddbm_configs["beta_d"], beta_min=ddbm_configs["beta_min"])
        else:
            raise NotImplementedError(f"pred_mode {ddbm_configs.get('pred_mode', 'vp')} not implemented for dbim sampler")
        ts = noise_schedule
        x_l, path, nfe, pred_x0, ts, first_noise = sample_dbim(denoise_net,
                x_T,
                ts,
                noise_schedule_obj,
                t_max=ddbm_configs["sigma_max"],
                eta=ddbm_configs.get("eta", 1.0),
                seed=ddbm_configs.get("seed", 42),
                )
    else:
        x_l, path, nfe = sample_heun(denoise_net,
                x_T,
                noise_schedule,
                pred_mode=ddbm_configs["pred_mode"],
                progress=False,
                sigma_max=ddbm_configs["sigma_max"],
                beta_d=ddbm_configs["beta_d"],
                beta_min=ddbm_configs["beta_min"],
                churn_step_ratio=ddbm_configs.get("churn_step_ratio", 0.33),
                guidance=ddbm_configs.get("guidance", 1),
                )
    # if diffusion_chunk_size is None:
        # x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    # else:
    #     x_l = []
    #     no_chunks = N_sample // diffusion_chunk_size + (
    #         N_sample % diffusion_chunk_size != 0
    #     )
    #     for i in range(no_chunks):
    #         chunk_n_sample = (
    #             diffusion_chunk_size
    #             if i < no_chunks - 1
    #             else N_sample - i * diffusion_chunk_size
    #         )
    #         chunk_x_l = _chunk_sample_diffusion(
    #             chunk_n_sample, inplace_safe=inplace_safe
    #         )
    #         x_l.append(chunk_x_l)
    #     x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]
    return x_l

def vp_logsnr(t, beta_d, beta_min):
    t = torch.as_tensor(t)
    return - torch.log((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1)
    
def vp_logs(t, beta_d, beta_min):
    t = torch.as_tensor(t)
    return -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min


def kabsch_torch_batched(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD, in a batched manner.
    :param P: A BxNx3 matrix of points
    :param Q: A BxNx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=1, keepdims=True)  # Bx1x3
    centroid_Q = torch.mean(Q, dim=1, keepdims=True)  #

    # Optimal translation
    t = centroid_Q - centroid_P  # Bx1x3
    t = t.squeeze(1)  # Bx3

    # Center the points
    p = P - centroid_P  # BxNx3
    q = Q - centroid_Q  # BxNx3

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(1, 2), q)  # Bx3x3

    # SVD
    U, S, Vt = torch.linalg.svd(H.float())  # Bx3x3

    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)).float())  # B
    flip = d < 0.0
    if flip.any().item():
        Vt[flip, -1, :] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    # RMSD
    rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(1, 2)) - q), dim=(1, 2)) / P.shape[1])

    # return R, t, rmsd
    return torch.matmul(p, R.transpose(1, 2)), q, rmsd

def kabsch_torch_batched_protein(P, Q, ligand_mask):
    """
    Batched Kabsch alignment using ONLY protein atoms (mask=0),
    but applies the rotation to ALL atoms (including ligand).

    Args:
        P: [B, N, 3] source coordinates (apo)
        Q: [B, N, 3] target coordinates (gt)
        ligand_mask: [N, 1] or [N], 1=ligand, 0=protein

    Returns:
        P_aligned: [B, N, 3] aligned P (all atoms)
        Q_centered: [B, N, 3] centered Q (all atoms)
        rmsd: [B] RMSD over protein atoms only
    """
    assert P.shape == Q.shape
    B, N, _ = P.shape
    device = P.device

    # ---- mask handling ----
    if ligand_mask.dim() == 2:
        ligand_mask = ligand_mask.squeeze(-1)  # [N]

    protein_mask = (ligand_mask == 0).to(P.dtype)  # [N]
    protein_mask = protein_mask.view(1, N, 1)      # [1, N, 1]
    protein_mask = protein_mask.to(device)

    # number of protein atoms
    num_protein = protein_mask.sum(dim=1, keepdim=True)  # [1,1,1]

    # ---- masked centroids (protein only) ----
    centroid_P = (P * protein_mask).sum(dim=1, keepdim=True) / num_protein
    centroid_Q = (Q * protein_mask).sum(dim=1, keepdim=True) / num_protein

    # ---- center all atoms ----
    p = P - centroid_P
    q = Q - centroid_Q

    # ---- use protein atoms only to compute H ----
    p_prot = p * protein_mask
    q_prot = q * protein_mask

    H = torch.matmul(p_prot.transpose(1, 2), q_prot)  # [B,3,3]

    # ---- SVD ----
    U, S, Vt = torch.linalg.svd(H.float())

    # ---- right-handed correction ----
    d = torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2))
    flip = d < 0
    if flip.any():
        Vt[flip, -1, :] *= -1.0

    R = Vt.transpose(1, 2) @ U.transpose(1, 2)  # [B,3,3]

    # ---- apply rotation to ALL atoms ----
    P_aligned = torch.matmul(p, R.transpose(1, 2))
    Q_centered = q

    # ---- RMSD (protein only) ----
    diff2 = ((P_aligned - Q_centered) ** 2) * protein_mask
    rmsd = torch.sqrt(
        diff2.sum(dim=(1, 2)) / num_protein.squeeze()
    )
    
    # put the ligand to the center
    ligand_mask = ligand_mask.to(bool)
    mask = ligand_mask.view(1, -1, 1)
    ligand_coords = P_aligned * mask
    ligand_centroid = ligand_coords.sum(dim=1, keepdim=True) / mask.sum()
    
    P_centered = P_aligned.clone()
    P_centered[mask.expand_as(P_aligned)] -= ligand_centroid.expand_as(P_aligned)[
        mask.expand_as(P_aligned)
    ]
    
    return P_centered, Q_centered, rmsd



def get_bridge_scalings(ddbm_configs, sigma):
    if ddbm_configs["pred_mode"] == 've':
        A = (
            sigma**4 / ddbm_configs["sigma_max"]**4 * ddbm_configs["sigma_data"]**2
            + (1 - sigma**2 / ddbm_configs["sigma_max"]**2)**2 * ddbm_configs["sigma_data"]**2
            + 2 * sigma**2 / ddbm_configs["sigma_max"]**2
            * (1 - sigma**2 / ddbm_configs["sigma_max"]**2)
            * ddbm_configs["cov_xy"]
            + ddbm_configs["c"]**2 * sigma**2 * (1 - sigma**2 / ddbm_configs["sigma_max"]**2)
        )

        c_in = 1 / (A)**0.5

        c_skip = (
            (1 - sigma**2 / ddbm_configs["sigma_max"]**2) * ddbm_configs["sigma_data"]**2
            + sigma**2 / ddbm_configs["sigma_max"]**2 * ddbm_configs["cov_xy"]
        ) / A

        c_out = (
            (sigma / ddbm_configs["sigma_max"])**4
            * (ddbm_configs["sigma_data"]**2 * ddbm_configs["sigma_data"]**2
            - ddbm_configs["cov_xy"]**2)
            + ddbm_configs["sigma_data"]**2
            * ddbm_configs["c"]**2
            * sigma**2
            * (1 - sigma**2 / ddbm_configs["sigma_max"]**2)
        )**0.5 * c_in

        return c_skip, c_out, c_in

    elif ddbm_configs["pred_mode"] == 'vp':
        logsnr_t = vp_logsnr(
            sigma,
            ddbm_configs["beta_d"],
            ddbm_configs["beta_min"],
        )
        logsnr_T = vp_logsnr(
            1,
            ddbm_configs["beta_d"],
            ddbm_configs["beta_min"],
        )

        logs_t = vp_logs(
            sigma,
            ddbm_configs["beta_d"],
            ddbm_configs["beta_min"],
        )
        logs_T = vp_logs(
            1,
            ddbm_configs["beta_d"],
            ddbm_configs["beta_min"],
        )

        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        c_t = -torch.expm1(logsnr_T - logsnr_t) * (2 * logs_t - logsnr_t).exp()

        A = (
            a_t**2 * ddbm_configs["sigma_data_end"]**2
            + b_t**2 * ddbm_configs["sigma_data"]**2
            + 2 * a_t * b_t * ddbm_configs["cov_xy"]
            + ddbm_configs["c"]**2 * c_t
        )

        c_in = 1 / (A)**0.5

        c_skip = (
            b_t * ddbm_configs["sigma_data"]**2
            + a_t * ddbm_configs["cov_xy"]
        ) / A

        c_out = (
            a_t**2
            * (ddbm_configs["sigma_data_end"]**2 * ddbm_configs["sigma_data"]**2
            - ddbm_configs["cov_xy"]**2)
            + ddbm_configs["sigma_data"]**2
            * ddbm_configs["c"]**2
            * c_t
        )**0.5 * c_in

        return c_skip, c_out, c_in


def sample_diffusion_training_ddbm(
    noise_sampler: TrainingNoiseSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
    use_conditioning: bool = True,
    ddbm_configs: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, ...]:
    """Implements diffusion training as ddbm discribed in the paper:Denoising Diffusion Bridge Models.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        label_dict (dict, optional) : a dictionary containing the followings.
            "coordinate": the ground-truth coordinates
                [..., N_atom, 3]
            "coordinate_mask": whether true coordinates exist.
                [..., N_atom]
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        N_sample (int): number of training samples
    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    batch_size_shape = label_dict["coordinate"].shape[:-2]
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype
    # Areate N_sample versions of the input structure by randomly rotating and translating
    x_gt_augment = centre_random_augmentation(
        x_input_coords=label_dict["coordinate"],
        N_sample=N_sample,
        mask=label_dict["coordinate_mask"],
    ).to(
        dtype
    )  # [..., N_sample, N_atom, 3]
    # TODO rigid align the apo structure to the x_gt_augment
    apo_atom_array = input_feature_dict['apo_atom_array']  # [..., N_atom, 3]
    
    # repeat the atom_array N_sample times
    apo_atom_array = apo_atom_array.unsqueeze(-3).expand(*batch_size_shape, N_sample, apo_atom_array.size(-2), 3)
    
    # import math
    
    # theta = matorch.pi / 4
    # R_gt = torch.tensor([
    #     [matorch.cos(theta), -matorch.sin(theta), 0.0],
    #     [matorch.sin(theta),  matorch.cos(theta), 0.0],
    #     [0.0,             0.0,             1.0],
    # ], device=device, dtype=dtype)

    # # 3. 已知平移
    # t_gt = torch.tensor([1.2, -0.7, 2.5], device=device, dtype=dtype)

    # # 4. 生成 Q
    # x_gt_augment = apo_atom_array @ R_gt.T + t_gt
    
    
    # R, t, rmsd = kabsch_torch_batched(apo_atom_array, x_gt_augment)
    # x_start = apo_atom_array @ R.transpose(-1, -2) + t.unsqueeze(-2)
    is_ligand_mask = input_feature_dict['is_ligand']
    x_start1, x_gt_augment1, rmsd = kabsch_torch_batched(apo_atom_array, x_gt_augment)
    x_start, x_gt_augment, rmsd2 = kabsch_torch_batched_protein(apo_atom_array, x_gt_augment, is_ligand_mask.to(bool))
    input_feature_dict['apo_atom_array'] = x_start
    
    new_rmsd = torch.sqrt(torch.sum((x_start - x_gt_augment)**2, dim=(-1, -2)) / x_gt_augment.size(-2))
    # TODO test the kabsch_torch_batched function
    
    # Add independent noise to each structure
    # sigma: independent noise-level [..., N_sample]
    # sigma = noise_sampler(size=(*batch_size_shape, N_sample), device=device).to(dtype)
    t, weights = noise_sampler.sample(N_sample, device=device)
    sigmas =torch.minimum(t, torch.ones_like(t)* ddbm_configs['sigma_max'])
    # noise: [..., N_sample, N_atom, 3]
    # noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma[..., None, None]
    noise = torch.randn_like(x_gt_augment, dtype=dtype)
    
    
    def bridge_sample(x0, xT, t):
        t = t.view(t.shape + (1,) * (x0.dim() - t.dim()))
        # t = append_dims(t, dims)
        # std_t = torch.sqrt(t)* torch.sqrt(1 - t / self.sigma_max)
        if ddbm_configs["pred_mode"].startswith('ve'):
            std_t = t* torch.sqrt(1 - t**2 / ddbm_configs['sigma_max']**2)
            mu_t= t**2 / ddbm_configs['sigma_max']**2 * xT + (1 - t**2 / ddbm_configs['sigma_max']**2) * x0
            samples = (mu_t +  std_t * noise )
        elif ddbm_configs["pred_mode"].startswith('vp'):
            logsnr_t = vp_logsnr(t, ddbm_configs['beta_d'], ddbm_configs['beta_min'])
            logsnr_T = vp_logsnr(ddbm_configs['sigma_max'], ddbm_configs['beta_d'], ddbm_configs['beta_min'])
            logs_t = vp_logs(t, ddbm_configs['beta_d'], ddbm_configs['beta_min'])
            logs_T = vp_logs(ddbm_configs['sigma_max'], ddbm_configs['beta_d'], ddbm_configs['beta_min'])

            a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
            b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()
            
            samples= a_t * xT + b_t * x0 + std_t * noise
            
        return samples
    
        
    x_noisy = bridge_sample(x_start, x_gt_augment, sigmas)

    c_skip, c_out, c_in = get_bridge_scalings(ddbm_configs, sigmas)
    
    # expand dim
    c_skip = c_skip.view(c_skip.shape + (1,) * (x_noisy.dim() - c_skip.dim()))
    c_out = c_out.view(c_out.shape + (1,) * (x_noisy.dim() - c_out.dim()))
    c_in = c_in.view(c_in.shape + (1,) * (x_noisy.dim() - c_in.dim()))

    weights = get_weightings(ddbm_configs, sigmas)
    if weights.max() > 10000:
        print(f'warning, weighted max is {weights.max().item()}')
    
    # weights = weights / weights.max()
    # Get denoising outputs [..., N_sample, N_atom, 3]
    diffusion_chunk_size = None
    if diffusion_chunk_size is None:
        x_denoised = denoise_net(
            x_noisy=x_noisy,
            t_hat_noise_level=sigmas,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            use_conditioning=use_conditioning,
            c_in=c_in,
            c_skip=c_skip,
            c_out=c_out,
        )
    else:
        x_denoised = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = (x_gt_augment + noise)[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            t_hat_noise_level_i = sigma[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]
            x_denoised_i = denoise_net(
                x_noisy=x_noisy_i,
                t_hat_noise_level=t_hat_noise_level_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                use_conditioning=use_conditioning,
            )
            x_denoised.append(x_denoised_i)
        x_denoised = torch.cat(x_denoised, dim=-3)

    return x_gt_augment, x_denoised, sigmas, weights

def get_snr(ddbm_configs, sigmas):
    if ddbm_configs["pred_mode"].startswith('vp'):
        return vp_logsnr(sigmas, ddbm_configs['beta_d'], ddbm_configs['beta_min']).exp()
    else:
        return sigmas**-2

def get_weightings(ddbm_configs, sigma):
    snrs = get_snr(ddbm_configs, sigma)

    if ddbm_configs["weight_schedule"] == "snr":
        weightings = snrs

    elif ddbm_configs["weight_schedule"] == "snr+1":
        weightings = snrs + 1

    elif ddbm_configs["weight_schedule"] == "karras":
        weightings = snrs + 1.0 / ddbm_configs["sigma_data"]**2

    elif ddbm_configs["weight_schedule"].startswith("bridge_karras"):
        if ddbm_configs["pred_mode"] == "ve":
            A = (
                sigma**4 / ddbm_configs["sigma_max"]**4 * ddbm_configs["sigma_data"]**2
                + (1 - sigma**2 / ddbm_configs["sigma_max"]**2)**2 * ddbm_configs["sigma_data"]**2
                + 2 * sigma**2 / ddbm_configs["sigma_max"]**2
                  * (1 - sigma**2 / ddbm_configs["sigma_max"]**2)
                  * ddbm_configs["cov_xy"]
                + ddbm_configs["c"]**2
                  * sigma**2
                  * (1 - sigma**2 / ddbm_configs["sigma_max"]**2)
            )

            weightings = A / (
                (sigma / ddbm_configs["sigma_max"])**4
                * (ddbm_configs["sigma_data"]**2 * ddbm_configs["sigma_data"]**2
                   - ddbm_configs["cov_xy"]**2)
                + ddbm_configs["sigma_data"]**2
                  * ddbm_configs["c"]**2
                  * sigma**2
                  * (1 - sigma**2 / ddbm_configs["sigma_max"]**2)
            )

        elif ddbm_configs["pred_mode"] == "vp":
            logsnr_t = vp_logsnr(sigma, ddbm_configs["beta_d"], ddbm_configs["beta_min"])
            logsnr_T = vp_logsnr(1, ddbm_configs["beta_d"], ddbm_configs["beta_min"])
            logs_t = vp_logs(sigma, ddbm_configs["beta_d"], ddbm_configs["beta_min"])
            logs_T = vp_logs(1, ddbm_configs["beta_d"], ddbm_configs["beta_min"])

            a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
            b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            c_t = -torch.expm1(logsnr_T - logsnr_t) * (2 * logs_t - logsnr_t).exp()

            A = (
                a_t**2 * ddbm_configs["sigma_data_end"]**2
                + b_t**2 * ddbm_configs["sigma_data"]**2
                + 2 * a_t * b_t * ddbm_configs["cov_xy"]
                + ddbm_configs["c"]**2 * c_t
            )

            weightings = A / (
                a_t**2 * (
                    ddbm_configs["sigma_data_end"]**2 * ddbm_configs["sigma_data"]**2
                    - ddbm_configs["cov_xy"]**2
                )
                + ddbm_configs["sigma_data"]**2 * ddbm_configs["c"]**2 * c_t
            )

        elif ddbm_configs["pred_mode"] in ["vp_simple", "ve_simple"]:
            weightings = torch.ones_like(snrs)

    elif ddbm_configs["weight_schedule"] == "truncated-snr":
        weightings = torch.clamp(snrs, min=1.0)

    elif ddbm_configs["weight_schedule"] == "uniform":
        weightings = torch.ones_like(snrs)

    else:
        raise NotImplementedError()

    return weightings



def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
    use_conditioning: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Implements diffusion training as described in AF3 Appendix at page 23.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        label_dict (dict, optional) : a dictionary containing the followings.
            "coordinate": the ground-truth coordinates
                [..., N_atom, 3]
            "coordinate_mask": whether true coordinates exist.
                [..., N_atom]
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        N_sample (int): number of training samples
    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    batch_size_shape = label_dict["coordinate"].shape[:-2]
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype
    # Areate N_sample versions of the input structure by randomly rotating and translating
    x_gt_augment = centre_random_augmentation(
        x_input_coords=label_dict["coordinate"],
        N_sample=N_sample,
        mask=label_dict["coordinate_mask"],
    ).to(
        dtype
    )  # [..., N_sample, N_atom, 3]

    # align the 'apo atom array' with the gt
    if 'apo_atom_array' in input_feature_dict:
        apo_atom_array = input_feature_dict['apo_atom_array']  # [..., N_atom, 3]
        # repeat the atom_array N_sample times
        apo_atom_array = apo_atom_array.unsqueeze(-3).expand(*batch_size_shape, N_sample, apo_atom_array.size(-2), 3)
        # R, t, rmsd = kabsch_torch_batched(apo_atom_array, x_gt_augment)
        # x_start = apo_atom_array @ R.transpose(-1, -2) + t.unsqueeze(-2)
        x_start, x_gt_augment, rmsd = kabsch_torch_batched(apo_atom_array, x_gt_augment)
        input_feature_dict['apo_atom_array'] = x_start
    
    
    # Add independent noise to each structure
    # sigma: independent noise-level [..., N_sample]
    sigma = noise_sampler(size=(*batch_size_shape, N_sample), device=device).to(dtype)
    # noise: [..., N_sample, N_atom, 3]
    noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma[..., None, None]

    # Get denoising outputs [..., N_sample, N_atom, 3]
    diffusion_chunk_size = None
    if diffusion_chunk_size is None:
        x_denoised = denoise_net(
            x_noisy=x_gt_augment + noise,
            t_hat_noise_level=sigma,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            use_conditioning=use_conditioning,
        )
    else:
        x_denoised = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = (x_gt_augment + noise)[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            t_hat_noise_level_i = sigma[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]
            x_denoised_i = denoise_net(
                x_noisy=x_noisy_i,
                t_hat_noise_level=t_hat_noise_level_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                use_conditioning=use_conditioning,
            )
            x_denoised.append(x_denoised_i)
        x_denoised = torch.cat(x_denoised, dim=-3)

    return x_gt_augment, x_denoised, sigma
