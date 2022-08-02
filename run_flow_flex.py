import numpy as np

import torch
from torch import nn
from torch import optim

import tqdm

import tkinter

import copy

import sys
sys.path.append('..')

import pyro
from samplers import mala, i_sir, ex2_mcmc

from cifar10_experiments.models import Generator, Discriminator

from sampling_utils.adaptive_mc import CISIR, Ex2MCMC, FlowMCMC
from sampling_utils.adaptive_sir_loss import MixKLLoss
from sampling_utils.distributions import (
    Banana,
    CauchyMixture,
    Distribution,
    Funnel,
    HalfBanana,
    IndependentNormal,
)
from sampling_utils.ebm_sampling import MALA
from sampling_utils.flows import RNVP
from sampling_utils.metrics import ESS, acl_spectrum
from sampling_utils.total_variation import (
    average_total_variation,
)


device = 'cpu'
lat_size=100
gen_cifar10 = Generator(lat_size)
gen_cifar10.to(device)

discr_cifar10 = Discriminator()
discr_cifar10.to(device)

prior_cifar10 = torch.distributions.MultivariateNormal(torch.zeros(lat_size).to(device), torch.eye(lat_size).to(device))


gen_cifar10.load_state_dict(torch.load('cifar10_experiments/weights/WGAN/generator.pkl', map_location='cpu'))
discr_cifar10.load_state_dict(torch.load('cifar10_experiments/weights/WGAN/discriminator.pkl', map_location='cpu'))
gen_cifar10.eval()
discr_cifar10.eval()
print('gans loaded')
def get_energy_wgan(z):
    return (-discr_cifar10(gen_cifar10(z)).squeeze() - prior_cifar10.log_prob(z).squeeze())

def log_target_dens(x):
    """
    returns the value of a target density - mixture of the 3 gaussians 
    """
    x = torch.FloatTensor(x).to(device)
    return -get_energy_wgan(x).detach().cpu().numpy()

def grad_log_target_dens(x):
    """
    returns the gradient of log-density 
    """
    x = torch.FloatTensor(x).to(device)
    x.requires_grad_(True)
    external_grad = torch.ones(x.shape[0])
    (-get_energy_wgan(x)).backward(gradient=external_grad)
    return x.grad.data.detach().cpu().numpy()


class distr:
    """
    Base class for a custom target distribution
    """

    def __init__(self, beta = 1.0):
        super().__init__()
        self.beta = beta

    def log_prob(self, z):
        """
        The method returns target logdensity, estimated at point z
        Input:
        z - datapoint
        Output:
        log_density: log p(z)
        """
        # You should define the class for your custom distribution
        return -get_energy_wgan(z).unsqueeze(0)

    def energy(self, z):
        """
        The method returns target logdensity, estimated at point z
        Input:
        z - datapoint
        Output:
        energy = -log p(z)
        """
        # You should define the class for your custom distribution
        return -get_energy_wgan(z).unsqueeze(0)

    def __call__(self, z):
        return self.log_prob(z)

params_flex = {
      "N": 5000,
      "grad_step": 0.2,
      "adapt_stepsize": False,
      "corr_coef": 0.0,
      "bernoulli_prob_corr": 0.0,
      "mala_steps": 0,
    "flow": {
      "num_flows": 5, # number of normalizing layers 
      "lr": 1e-3, # learning rate 
      "batch_size": 100,
      "n_steps": 50,
    }
}

beta = 1.0
scale_proposal = 1.0

target = distr(beta)

loc_proposal = torch.zeros(lat_size).to(device)
scale_proposal = scale_proposal * torch.ones(lat_size).to(device)

proposal = IndependentNormal(
    dim=lat_size,
    loc=loc_proposal,
    scale=scale_proposal,
    device=device,
)

#pyro.set_rng_seed(42)
pyro.set_rng_seed(142)
mcmc = Ex2MCMC(**params_flex, dim=lat_size)
verbose = mcmc.verbose
mcmc.verbose = False
flow = RNVP(params_flex["flow"]["num_flows"], dim=lat_size)
flow_mcmc = FlowMCMC(
    target,
    proposal,
    device,
    flow,
    mcmc,
    batch_size = params_flex["flow"]["batch_size"],
    lr=params_flex["flow"]["lr"],
)
flow.train()
out_samples, nll = flow_mcmc.train(
    n_steps=params_flex["flow"]["n_steps"],
)
print('out_samples',len(out_samples))
#print(torch.stack(out_samples).shape)
print('out_samples',out_samples[0].shape)

torch.save(torch.stack(out_samples), 'out_sample_flex_142_train_flows5_50iters.npy')
print(nll)

assert not torch.isnan(
    next(flow.parameters())[0, 0],
).item()

flow.eval()
mcmc.flow = flow
mcmc.verbose = verbose


n_steps_flex2 = 1000
N_traj = 200
mcmc.N = 5
mcmc.mala_steps = 0
mcmc.grad_step = 1e-6
all_res = []
for i in range(N_traj):
    pyro.set_rng_seed(342+i)
    start = proposal.sample((1,))
    # s = time.time()
    out = mcmc(start, target, proposal, n_steps = n_steps_flex2)
    print(out[1])
    if isinstance(out, tuple):
        sample = out[0]
    else:
        sample = out
    sample = np.array(
        [_.detach().numpy() for _ in sample],
    ).reshape(-1, 1, lat_size)
    all_res.append(copy.deepcopy(sample))
all_res = np.asarray(all_res)
print('all_res_142_numflows5',all_res.shape)
torch.save(all_res,'flex_allres_142_numflows5.npy')
