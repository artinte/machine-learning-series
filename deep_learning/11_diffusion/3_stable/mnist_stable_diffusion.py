import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch

# residual block
class ResBlock(nn.Module):
    def __init__(self, in_channel, time_emb_dim, out_channel=None,):
        super().__init__()
        if out_channel is None:
            out_channel = in_channel
        self.norm1 = nn.GroupNorm(32, in_channel, eps=1e-05, affine=True)

class UNet(nn.Module):
    '''
    A time-dependent score-based model built upon U-Net architecture.
    '''
    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        '''
        '''
        super().__init__()

    def forward(self, x, t, y=None):
        # Obtain the Gaussian random feature embedding for t.
        embed = self.act(self.time_embed(t))
        


num_steps = 500
def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           x_shape=(1, 28, 28),
                           num_steps=num_steps,
                           device='cuda',
                           eps=1e-3,
                           y=None):
    '''
    Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation of the
                            perturbation kernel.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps. Equivalent to the number of discreized
                    time steps.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.

    Returns:
        Samples generated.
    '''
    t = torch.ones(batch_size, device=device)


digit = 4
sample_batch_size = 64
num_steps = 250


sampler = Euler_Maruyama_sampler