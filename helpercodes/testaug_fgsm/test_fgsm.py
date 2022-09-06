import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from fgsm import FGSM
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from skimage.util import random_noise
import torchvision
import cv2
import torchvision.transforms as transforms
import numpy as np
import random
import os
import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

############ Model Architecture ##########
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) #in_channel x out_channel x kernel_size x stride
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

testset = torchvision.datasets.MNIST(
    root='./data/', train=False, download=True, transform=transform)

net = Net()
net = net.to(device)
net.load_state_dict(torch.load('./ckptw.pth')['net'])

net1 = Net()
net1 = net1.to(device)
#Change the filename here
model_loaded = (torch.load('/home/mt1/21CS60D06/MTP/wkg_code/AugMax/runs/MNIST/fat-1-untargeted-10-0.1_Lambda10.0_e200-b4000_sgd-lr0.1-m0.9-wd0.0005_cos/best_SA.pth'))
for key in list(model_loaded.keys()):
    model_loaded[key[7:]] = model_loaded.pop(key)

net1.load_state_dict(model_loaded)
# net1.load_state_dict(torch.load('../../runs/MNIST/fat-1-untargeted-10-0.1_Lambda10.0_e200-b4000_sgd-lr0.1-m0.9-wd0.0005_cos/best_SA.pth')['net'])


subset_indices_adv = np.load('../../indices/mnist_test_aug_indices.npy')
#np.save('mnist_test_aug_indices.npy',subset_indices_adv)
subset_test = torch.utils.data.Subset(testset, subset_indices_adv)
testloaderb = torch.utils.data.DataLoader(subset_test, batch_size=100, shuffle=False, num_workers=2)
print(len(testloaderb))


# Model
torch.manual_seed(123)
print('==> Building model..')
criterion = nn.CrossEntropyLoss()
    
def test(epoch,net):
    acc = []
    epv = []
    global best_acc
    net.eval()    
    #with torch.no_grad():
    for ep in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        test_loss = 0
        correct = 0
        total = 0
        atk = FGSM(net, eps=ep)
        for batch_idx, (inputs, targets) in enumerate(testloaderb):
            inputs, targets = inputs.to(device), targets.to(device)
            _,inputs = atk(inputs, targets)
            outputs = net1(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Save checkpoint.
        acc.append(100.*correct/total)
        epv.append(ep)
    print(acc)
    print(epv)

for epoch in range(1): #Run till convergence
    print("Testing")
    #test1(epoch)
    test(epoch,net)
    #scheduler.step()