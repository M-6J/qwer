import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import warnings
import os
from os import listdir
import numpy as np
import time
from os.path import isfile, join


class cifar10dvs(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))  # 48 48
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        # print(data.shape)
        # if self.train:
        new_data = []
        for t in range(data.size(-1)):
            new_data.append(self.tensorx(self.resize(self.imgx(data[..., t]))))
        data = torch.stack(new_data, dim=0)
        if self.transform is not None:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


def build_cifar10dvs(path):
    train_path = path + '/train'
    val_path = path + '/test'
    train_dataset = cifar10dvs(root=train_path, transform=False)
    val_dataset = cifar10dvs(root=val_path)

    return train_dataset, val_dataset