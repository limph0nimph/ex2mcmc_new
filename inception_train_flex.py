
from pathlib import Path
import numpy as np
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
from collections import defaultdict

from matplotlib import pyplot as plt
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



SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22 #18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("lines", linewidth=3)
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

device = 'cpu'
inception_model = inception_v3(
            pretrained=True,
            transform_input=False,  # False
            aux_logits=False,
).to(device)
inception_model.eval()

print('inception loaded')

N_INCEPTION_CLASSES = 1000
MEAN_TRASFORM = [0.485, 0.456, 0.406]
STD_TRANSFORM = [0.229, 0.224, 0.225]
N_GEN_IMAGES = 5000


def get_inception_score(
    imgs,
    inception_model,
    cuda: bool = True,
    batch_size: int = 32,
    resize: bool = False,
    splits: int = 1,
    device = 0,
):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N >= batch_size

    # Set up dataloader
    if not isinstance(imgs, torch.utils.data.Dataset):
        imgs = torch.utils.data.TensorDataset(imgs)

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    up = nn.Upsample(size=(299, 299), mode="bilinear").to(device)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, -1).data.cpu()

    # Get predictions
    preds = torch.zeros((N, N_INCEPTION_CLASSES))

    for i, batch in enumerate(dataloader):
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.to(device)
        batch_size_i = batch.size()[0]

        preds[i * batch_size : i * batch_size + batch_size_i] = get_pred(batch)
        
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        pis = preds[k * (N // splits) : (k + 1) * (N // splits), :]
        score = (
                (pis * (torch.log(pis) - torch.log(pis.mean(0)[None, :])))
                .sum(1)
                .mean(0)
            )
        score = torch.exp(score)
        split_scores.append(score)

    return (
        torch.mean(torch.stack(split_scores, 0)),
        torch.std(torch.stack(split_scores, 0)),
        preds,
    )


transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

N_traj=100
Traj_len=1000
sample_traj_flex = torch.load('out_sample_flex_42_train.npy', map_location=device)

sample_traj_flex = torch.as_tensor(sample_traj_flex.reshape(N_traj, Traj_len, lat_size))

sample_traj_flex2 = torch.load('out_sample_flex_142_train.npy', map_location=device)
sample_traj_flex2 = torch.as_tensor(sample_traj_flex2.reshape(100, Traj_len, lat_size))

sample_traj_flex = torch.cat([sample_traj_flex, sample_traj_flex2],dim = 0)

print('sample_traj_flex', sample_traj_flex.shape)
print('sample loaded')
arange = np.arange(0, 1000, 5)




mean_list = []
std_list = []
for step in arange:
    batch  = sample_traj_flex[:,step,:]
    batch = torch.FloatTensor(batch).to(device)
    imgs = gen_cifar10(batch)
    imgs = (imgs + 1) / 2
    imgs = transform(imgs)
    mean_score, _ = get_inception_score(imgs, inception_model=inception_model, device=device, resize=True,
        batch_size=80, splits=1)[:2]
    
    print('flex', 'score IS', mean_score, 'step = ', step)
    mean_list.append(mean_score)
    if step%50==0:
        np.save(f'IS_list_flex_train', np.array(mean_list))
    #std_list.append(std_score)
mean_list = np.array(mean_list)   
#std_list = np.array(std_list)  

line, = plt.plot(arange, 
mean_list,label = 'flex')
#plt.fill_between(arange, mean_list - 1.96 * std_list, mean_list + 1.96 * std_list, alpha=0.2, color=line.get_color())
np.save(f'IS_list_flex_train', mean_list)
    
plt.legend()
plt.ylabel('Inception Score Flex')
plt.xlabel('Iteration')
plt.savefig(f'inception_score_flex_1000iters_train.pdf')
plt.show()
