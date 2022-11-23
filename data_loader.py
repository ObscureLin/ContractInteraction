# data loader
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import numpy as np


# ==========================dataset load==========================
class TrainDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.masks = labels
        self.transforms = None

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.masks[idx]

        image = torch.from_numpy(image).float()
        label_numpy = np.array(label)
        label = torch.from_numpy(label_numpy).long()

        if self.transforms:
            image, label = self.transforms(image, label)

        sample = {'image': image, 'label': label}
        return sample
