from typing import Any
import pandas as pd
from datetime import datetime
from itertools import chain
import numpy as np
from torch.utils.data import DataLoader

from SkyImageDataset import SkyImageDataset

class PreprocessImgData():
    def __init__(self, L, H, step):
        # self.power = pd.read_json(power_file)
        # self.weather = pd.read_csv(weather_file)
        # self.df = pd.read_csv(data_path)
        # self.df = self.get_dataframe()

        self.L = L
        self.H = H
        self.step = step
        self.batch_size = 64

    def run(self) -> Any:
        trainset, trainloader = self.get_dataloader("train")
        return trainset, trainloader
    
    # for ghi column, and time covariates
    def generate_timeseries(self, data):
        n = self.L + self.H
        Y = np.array([data[i:i+n] for i in range(len(data)-n + self.step)])
        Yl = Y[:, :self.L] # lookback/input sequence
        Yh = Y[:, self.L:] # horizon/target sequence
        return Yl, Yh

    def generate_image_sequences(self, data):
        n = self.L + self.H
        sequences = np.array([data[i:i+n] for i in range(len(data)-n + self.step)])
        Xl = sequences[:, :self.L]
        return Xl
    
    def get_dataloader(self, flag):
        if flag == 'train':
            dataset = SkyImageDataset("../Datasets/images_train.csv")
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False) 
        elif flag == 'val':
            dataset = SkyImageDataset("../Datasets/images_val.csv")
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False) 
        else:
            dataset = SkyImageDataset("../Datasets/images_test.csv")
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False) 

        return dataset, dataloader
    
pipeline = PreprocessImgData(2,2,1)
dataset,dataloader = pipeline.run()
print(dataset.__getitem__(11133))
print(len(dataloader))