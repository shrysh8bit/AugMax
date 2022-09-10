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

def MNIST_FGSM_dataloader(num_classes=10, batch_size = 96):
    print('5. Loading dataset MNIST FGSM')
    
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    train_data = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)

    test_data = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
    return train_data, test_data
