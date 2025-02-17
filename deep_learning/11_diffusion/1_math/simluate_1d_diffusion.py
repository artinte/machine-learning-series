import numpy
from matplotlib import pyplot
from scipy.stats import norm

# Simulate SDE with drift function f and noise amplitude g for arbitrary number of steps.
def forward_SDE_simulation(x0, nsteps, dt, f, g, params):
    # Initialize time and a stochastic trajectory.
    t = 0
    x_traj = numpy.zeros((nsteps + 1, *x0.shape))
    x_traj[0] = numpy.copy(x0)

    rng = numpy.random.default_rng(0)
    # Proform many Euler-Maruyama time steps.
    for i in range(nsteps):
        random_normal = numpy.random.standard_normal(*x0.shape)

        x_traj[i+1] = x_traj[i] + f(x_traj[i], t, params) * dt + \
                g(x_traj[i], t, params)* numpy.sqrt(dt) * random_normal
        t = t + dt
    
    return x_traj


# Drift function for diffusion (returns zeros)
def f_diff_simple(x, t, params):
    return numpy.zeros((*x.shape,))

# Nosize amplitude for diffusion (constant)
def g_diff_simple(x, t, params):
    sigma = params['sigma']
    return sigma * numpy.ones((*x.shape,))

def transition_probability_diffusion_exact(x, t, params):
    x0, sigma = params['x0'], params['sigma']
    # pdf of normal distribution with mean x0 and variance (sigma^2)*t
    pdf = norm.pdf(x, loc=x0, scale=numpy.sqrt((sigma**2)*t))
    return pdf

sigma = 1   # noise amplitude for 1D diffusion

num_samples = 1000
x0 = numpy.zeros(num_samples)   # Initial condition for diffusion

nsteps = 2000   # number of simulation steps
dt = 0.001      # size of small time steps
T = nsteps * dt
t = numpy.linspace(0, T, nsteps + 1)

params = {'sigma': sigma, 'x0': x0, 'T': T}
x_traj = forward_SDE_simulation(x0, nsteps, dt, f_diff_simple, g_diff_simple, params)

# Plot initial distribution (distribution before diffusion).
pyplot.hist(x_traj[0], density=True, bins=100)
pyplot.title('$t = 0$')
pyplot.xlabel('$x$')
pyplot.ylabel('Probability')
pyplot.grid()
pyplot.show()

# Compute exact trainstion probability.
x_f_min, x_f_max = numpy.amin(x_traj[-1]), numpy.amax(x_traj[-1])
num_xf = 1000
x_f_arg = numpy.linspace(x_f_min, x_f_max, num_xf)
pdf_final = transition_probability_diffusion_exact(x_f_arg, T, params)

# Plot final distribution (distribution after diffusion).
pyplot.hist(x_traj[-1], bins=100, density=True)
pyplot.plot(x_f_arg, pdf_final, linewidth=4)
pyplot.title('$t = $'+str(T))
pyplot.xlabel('$x$')
pyplot.ylabel('Probability')
pyplot.grid()
pyplot.show()

# Plot some trajectories.
sample_trajectories = [0, 1, 2, 3, 4]
for s in sample_trajectories:
    pyplot.plot(t, x_traj[:, s])
pyplot.title('Sample trajectoies')
pyplot.xlabel('$t$')
pyplot.ylabel('x')
pyplot.grid()
pyplot.show()
