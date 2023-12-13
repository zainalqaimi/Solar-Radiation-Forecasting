import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import time
import random

import datetime
from sklearn.preprocessing import StandardScaler

from Dataloader.ImageDataset import ImageDataset
from Model.ResNet import ResNet

from Pipeline import Pipeline

fix_seed = 7
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

transform = 'rescale'

batch_size = 64
lr = 0.01
epochs = 30

model_type = 'predict'
L = 5
H = 5
step = 5
output_dim = 1024
hidden_dim = 16
blocks = [4,8,12]

for block in blocks:
    setting = 'ResNetRegr-{}_L{}_H{}_step{}_outdim{}_hidden{}_epochs{}_bs{}_lr{}_transform{}'.format(
                    block,
                    L,
                    H,
                    step,
                    output_dim,
                    hidden_dim,
                    epochs,
                    batch_size,
                    lr,
                    transform)

    print('-------- Starting Experiment : {} --------'.format(setting))
    pipeline = Pipeline(model_type, L, H, output_dim, hidden_dim, step, batch_size, lr, epochs, setting, 
                    block, transform, device)

    train_loader, vali_loader, test_loader = pipeline.get_dataloader()
    # print(len(train_loader), len(vali_loader), len(test_loader))

    # pipeline.train(train_loader, vali_loader, test_loader)
    # pipeline.test(test_loader)
    torch.cuda.empty_cache()


