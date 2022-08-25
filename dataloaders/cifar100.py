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

def build_cifar100(cutout=False, use_cifar10=True, download=False):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root='./raw/',
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root='./raw/',
                              train=False, download=download, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='./raw/',
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root='./raw/',
                               train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset