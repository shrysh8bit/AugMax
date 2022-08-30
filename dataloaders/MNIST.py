import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# from fgsm import FGSM
import torchvision
from torchvision import datasets
from torchvision import transforms
import numpy as np
import random
import os
import argparse

import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np 
from PIL import Image

def MNIST_dataloaders(data_dir, num_classes=10, AugMax=None, batch_size = 96, **AugMax_args):
    print('==> Loading data..MNIST')
    
    # transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    # train_data = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    # val_set = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)

    # transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    # MNIST = datasets.MNIST

    # trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    # valset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)


    # print(f"train loader mnist len {len(trainloader)}")
    # assert num_classes in [10, 100]
    # CIFAR = datasets.CIFAR10 if num_classes == 10 else datasets.CIFAR100

    if AugMax is not None:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4)])
        test_transform = transforms.Compose([transforms.ToTensor()])

        train_data = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=train_transform)
        test_data = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=test_transform)
        
        train_data = AugMax(train_data, test_transform, 
            mixture_width=AugMax_args['mixture_width'], mixture_depth=AugMax_args['mixture_depth'], aug_severity=AugMax_args['aug_severity'], 
        )

    else:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor()])
        test_transform = transforms.Compose(
            [transforms.ToTensor()])

        train_data = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=train_transform)
        test_data = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=test_transform)
        
    # train_data = MNIST(data_dir, train=True, transform=train_transform, download=True)
    # test_data = MNIST(data_dir, train=False, transform=test_transform, download=True)
    
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=False, num_workers=2)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=False, num_workers=2)
    # train_data = torch.utils.data.DataLoader(trainset, batch_size, shuffle=False, num_workers=2)
    # test_data = torch.utils.data.DataLoader(valset, batch_size, shuffle=False, num_workers=2)
    
    return train_data, test_data
