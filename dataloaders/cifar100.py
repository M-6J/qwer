import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

def build_cifar100(download=False):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())

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
    train_dataset = CIFAR100(root='./dataset/',
                                train=True, download=download, transform=transform_train)
    val_dataset = CIFAR100(root='./dataset/',
                            train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset