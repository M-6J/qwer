import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def build_mnist(download=False):
    train_dataset = MNIST(root='./dataset/',
                             train=True, download=download, transform=transforms.ToTensor())
    val_dataset = MNIST(root='./dataset/',
                           train=False, download=download, transform=transforms.ToTensor())
    return train_dataset, val_dataset