import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# from fgsm import FGSM
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os
import argparse

def MNIST_dataloaders(data_dir, num_classes=10, AugMax=None, batch_size = 96, **AugMax_args):
    print('==> Preparing data..MNIST')
    
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    valset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(valset, batch_size, shuffle=False, num_workers=2)

    # print(f"train loader mnist len {len(trainloader)}")

    # if AugMax is not None:
    #     train_transform = transforms.Compose(
    #         [transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, padding=4)])
    #     test_transform = transforms.Compose([transforms.ToTensor()])

    #     train_data = CIFAR(data_dir, train=True, transform=train_transform, download=True)
    #     test_data = CIFAR(data_dir, train=False, transform=test_transform, download=True)
    #     train_data = AugMax(train_data, test_transform, 
    #         mixture_width=AugMax_args['mixture_width'], mixture_depth=AugMax_args['mixture_depth'], aug_severity=AugMax_args['aug_severity'], 
    #     )

    # else:
    #     train_transform = transforms.Compose(
    #         [transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.ToTensor()])
    #     test_transform = transforms.Compose(
    #         [transforms.ToTensor()])

    #     train_data = CIFAR(data_dir, train=True, transform=train_transform, download=True)
    #     test_data = CIFAR(data_dir, train=False, transform=test_transform, download=True)
    
    return train_loader, val_loader