import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST
import warnings
import os
import torchvision
from os import listdir
import numpy as np
import time
from os.path import isfile, join

def build_imagenet():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    root = '/data_smr/dataset/ImageNet'
    train_root = os.path.join(root,'train')
    val_root = os.path.join(root,'val')
    train_dataset = ImageFolder(
        train_root,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_dataset = ImageFolder(
        val_root,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    return train_dataset, val_dataset

