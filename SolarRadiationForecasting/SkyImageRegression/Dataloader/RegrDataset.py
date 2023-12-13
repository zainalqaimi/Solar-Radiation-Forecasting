import torch
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from torchvision import transforms

class RegrDataset(Dataset):
    def __init__(self, y, X_img, Xt, mean, std, transform=None):
        self.y = y
        self.X_img = X_img 
        self.Xt = Xt
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self):
        return len(self.y)


    def __getitem__(self, idx):
        # Extract x values (images and time variables)
        path = self.X_img[idx]

        # Read and normalize x images
        path = os.path.join('../', path)
        img = Image.open(path).convert('RGB')
        img = transforms.ToTensor()(img)
        if self.transform:
            img = self.transform(img)

        return (img, self.Xt[idx]), self.y[idx]
