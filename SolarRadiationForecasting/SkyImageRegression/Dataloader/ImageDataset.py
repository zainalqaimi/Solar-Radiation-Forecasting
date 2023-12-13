import torch
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, yl, yh, X_img, Xt, mean, std, transform=None):
        self.yl = yl
        self.yh = yh
        self.X_img = X_img 
        self.Xt = Xt
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self):
        return len(self.yh)


    def __getitem__(self, idx):
        # Extract x values (images and time variables)
        x_image_paths = self.X_img[idx]

        # Read and normalize x images
        x_images = []
        for path in x_image_paths:
            path = os.path.join('../', path)
            img = Image.open(path).convert('RGB')
            img = transforms.ToTensor()(img)
            if self.transform:
                img = self.transform(img)
            x_images.append(img)

        # Convert x images to tensors
        # x_images = torch.stack([torch.tensor(image) for image in x_images])
        x_images = torch.stack(x_images)

        # Convert time variables to tensors
        # xt = torch.tensor(xt)

        # Convert y ghi values to tensor


        return (self.yl[idx], x_images, self.Xt[idx]), self.yh[idx]
