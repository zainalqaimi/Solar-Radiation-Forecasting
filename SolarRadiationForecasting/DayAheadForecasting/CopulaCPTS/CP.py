import torch
import torch.nn as nn
import numpy as np
# from copulae import GumbelCopula
# from copulae.core import pseudo_obs
import pandas as pd
from tqdm import trange
from scipy.stats import rankdata

def pseudo_obs(data):
    ranks = rankdata(data)  # Compute ranks of the data points, handling ties
    pseudo_data = ranks / (len(data) + 1)  # Convert ranks to pseudo-observations

    return pseudo_data


def gumbel_copula_loss(x, cop, data, epsilon):
    return np.fabs(cop.cdf([x] * data.shape[1]) - 1 + epsilon)


def empirical_copula_loss(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return np.fabs(np.mean(np.all(np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])), axis=1)
                           ) - 1 + epsilon)


def empirical_copula_loss_new(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return np.mean(np.all(np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])), axis=1)
                           ) - 1 + epsilon


class CP(nn.Module):
    def __init__(self, dimension, epsilon):
        super(CP, self).__init__()
        self.alphas = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon
        self.relu = torch.nn.ReLU()
    
    def forward(self, pseudo_data):
        coverage = torch.mean(torch.relu(torch.prod(torch.sigmoid((self.alphas-pseudo_data)*1000), dim=1)))
        return torch.abs(coverage-1+self.epsilon)



def search_alpha(alpha_input, epsilon, epochs=500):
    #pseudo_data = torch.tensor(pseudo_obs(alpha_input))
    pseudo_data = torch.tensor(alpha_input)
    dim = alpha_input.shape[-1]
    cp = CP(dim, epsilon)
    # loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(cp.parameters(),weight_decay=1e-4)

    with trange(epochs, desc="training", unit="epochs") as pbar:
        for i in pbar:
            optimizer.zero_grad()
            loss = cp(pseudo_data)
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.detach().numpy())

    return cp.alphas.detach().numpy()