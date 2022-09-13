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
    print('5. Loading dataset MNIST')
    
    if AugMax is not None:
        all_train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4)])
        test_transform = transforms.Compose([transforms.ToTensor()])

        all_train_data = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=all_train_transform)
        test_data = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=test_transform)
        
        print(f"6. Len of data: all train {len(all_train_data)}    test {len(test_data)}")
        
        train_aug_indices = np.load('indices/mnist_train_aug_indices.npy')
        train_non_aug_indices = np.load('indices/mnist_train_non_aug_indices.npy')

        clean_data = torch.utils.data.Subset(all_train_data, train_non_aug_indices)
        augmax_data =torch.utils.data.Subset(all_train_data, train_aug_indices)

        print(f"6. Len of data: clean {len(clean_data)}    augmax {len(augmax_data)}")
        print(f"6. type of augmax data {type(augmax_data)}")

        augmax_data = AugMax(augmax_data, test_transform, 
            mixture_width=AugMax_args['mixture_width'], mixture_depth=AugMax_args['mixture_depth'], aug_severity=AugMax_args['aug_severity'])
        
        clean_data = AugMax(clean_data, test_transform, 
            mixture_width=1, mixture_depth=0, aug_severity=AugMax_args['aug_severity'])

        print(f"7. type of augmax data {type(augmax_data)}")

        train_data = torch.utils.data.ConcatDataset([augmax_data, clean_data])

        print(f"7. Data loading complete")


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
