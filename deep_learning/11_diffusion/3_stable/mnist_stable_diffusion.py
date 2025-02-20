import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def forward_diffusion_1D(x0, noise_strength_fn, t0, nsteps, dt):
    '''
    x0: initial sample value, scalar
    noise_strength_fn: function of time, outputs scalar noise strength
    t0: initial time
    nsteps: number of diffusion steps
    dt: time step size
    '''
    # Initialize trajectory.
    x = np.zeros(nsteps+1)
    x[0] = x0
    t = t0 + np.arange(nsteps+1) * dt
    
    # Perform many Euler-Maruyama time steps.
    for i in range(nsteps):
        noise_strength = noise_strength_fn(t[i])
        random_normal = np.random.randn()
        x[i+1] = x[i] + np.sqrt(dt) * noise_strength * random_normal
    return x, t

# Example noise strength function: always equal to 1.
def noise_strength_constant(t):
    return 1

nsteps = 100
t0 = 0
dt = 0.1
noise_strength_fn = noise_strength_constant
x0 = 0
num_tries = 5
for i in range(num_tries):
    x, t = forward_diffusion_1D(x0, noise_strength_fn, t0, nsteps, dt)
    
    plt.plot(t, x)
    plt.xlabel('time')
    plt.ylabel('$x$')
plt.title('Forward diffusion visualized')
plt.show()

# Simulate forward diffusion for N steps.
def reverse_diffusion_1D(x0, noise_strength_fn, score_fn, T, nsteps, dt):
    '''
    x0: initial sample value, scalar
    noise_strength_fn: function of time, outputs scalar noise strength
    score_fn: score_function
    T: final time
    nsteps: number of diffusion steps
    dt: time step size
    '''
    # Initialize trajectory.
    x = np.zeros(nsteps+1)
    x[0] = x0
    t = np.arange(nsteps+1) * dt
    
    # Perform many Euler-Maruyama time steps.
    for i in range(nsteps):
        noise_strength = noise_strength_fn(T - t[i])
        score = score_fn(x[i], 0, noise_strength, T - t[i])
        random_normal = np.random.randn()
        x[i+1] = x[i] + (noise_strength**2)*score*dt + np.sqrt(dt) * noise_strength * random_normal
    return x, t

# Example noise strength function: always equal to 1.
def score_simple(x, x0, noise_strength, t):
    score = - (x - x0) / ((noise_strength**2)*t)
    return score

nsteps = 100
t0 = 0
dt = 0.1
noise_strength_fn = noise_strength_constant
score_fn = score_simple
x0 = 0
T = 11

num_tries = 5
for i in range(num_tries):
    # Draw from the noise distribution.
    x0 = np.random.normal(loc=0, scale=T)
    x, t = reverse_diffusion_1D(x0, noise_strength_fn, score_fn, T, nsteps, dt)
    plt.plot(t, x)
    plt.xlabel('time')
    plt.ylabel('$x$')
    plt.title('Reverse diffusion visualized')
plt.show()


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
