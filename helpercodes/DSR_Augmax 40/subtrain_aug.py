import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from subsetpack.fgsm import FGSM
import torchvision
import torchvision.transforms as transforms
from subsetpack.augmentations import augmentations
import numpy as np
import random
import copy
import os
import argparse
from PIL import Image


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()


class AugmentedDataset(torch.utils.data.Dataset):

  def __init__(self, X, y, scale_data=True):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)
      print(self.X.shape)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]


def aug(image, preprocess, mixture_depth, aug_severity):
	"""Perform augmentation operations on PIL.Images.
	Args:
		image: PIL.Image input image
		preprocess: Preprocessing function which should return a torch tensor.
	Returns:
		image_aug: Augmented image.
	"""
	#aug_list = augmentations.augmentations
	aug_list = augmentations
	image = image.squeeze(0)
	#print(image.shape)
	image_aug = copy.deepcopy(image)
	transform = transforms.ToPILImage()
	image_aug = transform(image_aug)
	depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
	for _ in range(depth):
		op = np.random.choice(aug_list)
		image_aug = op(image_aug, aug_severity)
		
	image_aug = preprocess(image_aug)

	return image_aug
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

trainset = torchvision.datasets.MNIST(
    root='./data/', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

trainl = torch.utils.data.DataLoader(trainset,shuffle=False,batch_size = 1)
ftrain = []
ytrain = []
random.seed(123)
test_transform = transforms.Compose([transforms.ToTensor()])
#trainind = np.load('./mnist_train_aug_indices.npy')
#print(trainind)
for batchid, (ip, targ) in enumerate(trainl):
    #if batchid in trainind:
    ftrain.append(ip.numpy())
    ytrain.append(targ.item())
    for _ in range(1):
        ftrain.append(np.expand_dims(aug(ip, test_transform, -1, 3).cpu().numpy(),axis=0))
        ytrain.append(targ.item())
    '''else:
        ftrain.append(ip.numpy())
        ytrain.append(targ.item())'''

trainset = AugmentedDataset(np.asarray(ftrain),np.asarray(ytrain))
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
#epbatch = [0, 262, 54, 3, 257, 256, 277, 219, 204, 37, 259, 26, 195, 258, 272, 224, 269, 205, 190, 19, 20, 21, 139, 14, 24, 226, 4, 192, 223, 29, 193, 183, 32, 33, 78, 57, 36, 182, 177, 228, 149, 64, 86, 170, 163, 45, 46, 47, 171, 180, 156, 239, 263, 53, 210, 30, 56, 89, 42, 59, 157, 174, 62, 261, 267, 22, 166, 67, 118, 148, 146, 15, 72, 100, 107, 265, 76, 209, 116, 132, 80, 81, 137, 83, 238, 141, 151, 87, 88, 188, 90, 91, 212, 12, 10, 95, 34, 203, 140, 99, 23, 134, 11, 220, 38, 9, 8, 131, 130, 2, 124, 111, 225, 122, 84, 216, 129, 120, 1, 147]
#epbatch = [0, 457, 196, 170, 247, 219, 38, 443, 383, 259, 232, 266, 441, 264, 249, 263, 262, 17, 18, 258, 20, 21, 22, 23, 24, 25, 26, 27, 183, 29, 30, 31, 32, 33, 34, 35, 36, 37, 9, 39, 158, 41, 3, 230, 44, 257, 46, 47, 186, 49, 50, 51, 52, 53, 265, 442, 2, 57, 160, 59, 155, 161, 185, 63, 6, 80, 13, 67, 68, 222, 70, 12, 72, 73, 171, 124, 5, 112, 78, 79, 8, 348, 174, 83, 84, 85, 151, 439, 1, 16, 132, 180, 178, 361, 94, 267, 96, 254, 138, 99, 15, 101, 136, 103, 149, 108, 10, 164, 244, 165, 146, 95, 208, 113, 179, 115, 426, 117, 167, 188, 120, 130, 143, 134, 7, 229, 305, 118, 128, 145, 111, 142]
#Augmax all augmented 20%
epbatch = [0, 1160, 59, 868, 62, 57, 35, 1007, 80, 14, 31, 1182, 75, 77, 55, 78, 73, 636, 18, 1058, 846, 234, 390, 23, 283, 479, 26, 260, 585, 328, 30, 513, 491, 304, 34, 952, 66, 37, 38, 369, 225, 889, 42, 728, 44, 45, 313, 572, 793, 563, 426, 51, 52, 53, 54, 695, 540, 719, 58, 244, 614, 61, 566, 420, 64, 969, 459, 67, 1048, 69, 747, 71, 573, 233, 74, 394, 76, 81, 33, 79, 1091, 243, 3, 594, 84, 886, 1097, 537, 88, 543, 521, 91, 92, 481, 506, 938, 96, 97, 536, 990, 100, 635, 102, 685, 99, 463, 165, 1198, 232, 715, 110, 546, 386, 994, 1047, 115, 24, 512, 1166, 292, 580, 206, 673, 123, 124, 125, 379, 316, 436, 129, 120, 149, 132, 352, 441, 309, 533, 1046, 407, 139, 140, 1090, 1083, 318, 382, 398, 288, 147, 391, 487, 1113, 529, 511, 456, 5, 488, 781, 1022, 111, 159, 577, 720, 1021, 1015, 943, 25, 501, 1177, 878, 425, 358, 1037, 305, 104, 1, 490, 176, 1170, 1051, 11, 987, 1061, 303, 183, 383, 646, 743, 637, 995, 1169, 1165, 2, 1054, 626, 971, 195, 1023, 1059, 340, 437, 362, 480, 444, 203, 204, 796, 1197, 1027, 881, 209, 765, 564, 778, 213, 214, 242, 957, 1116, 72, 907, 241, 535, 174, 254, 1142, 271, 991, 547, 438, 229, 1033, 1063, 1109, 968, 311, 582, 1084, 852, 1167, 239]
#epbatch = [0, 483, 299, 392, 390, 1, 496, 7, 329, 384, 8, 2, 3, 4, 5, 346, 481, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 123, 129, 30, 31, 32, 171, 34, 35, 36, 37, 38, 39, 488, 41, 42, 43, 146, 168, 46, 47, 48, 49, 50, 9, 52, 125, 435, 55, 56, 57, 58, 59, 420, 61, 147, 63, 484, 472, 66, 67, 68, 69, 364, 71, 126, 290, 74, 75, 76, 77, 134, 79, 80, 81, 82, 466, 84, 85, 86, 262, 88, 89, 90, 155, 122, 93, 152, 151, 418, 97, 98, 130, 100, 142, 195, 471, 393, 105, 106, 107, 108, 6, 480, 438, 469, 120, 114, 485, 141, 487, 462, 128]
subset_trindices = []


for elem in epbatch:
	batchid = elem
	#print(batchid)
	for i in range(batchid*100,(batchid*100)+100):
		subset_trindices.append(i)

print(len(set(subset_trindices)))

subset_train = torch.utils.data.Subset(trainset, subset_trindices)

trainloader = torch.utils.data.DataLoader(
    subset_train, batch_size=100, shuffle=True, num_workers=2)

print(len(trainloader))

testset = torchvision.datasets.MNIST(
    root='./data/', train=False, download=True, transform=transform)

testloaderb = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
torch.manual_seed(123)
print('==> Building model..')
net = Net()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)

#optimizer = optim.SGD(net.parameters(), lr=0.1)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.squeeze(1)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(train_loss/batch_idx)
    
def test(epoch):
    global best_acc
    net.eval()
    #net.load_state_dict(torch.load('./checkpoint_sub/ckpt_sub20_otheraug_all.pth')['net'])
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloaderb):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_sub'):
            os.mkdir('checkpoint_sub')
        torch.save(state, './checkpoint_sub/ckpt_sub20_otheraug_all.pth')
        best_acc = acc

    print(acc)
    print("Best Accuracy")
    print(best_acc)

for epoch in range(start_epoch,200): #Run till convergence
    print("Training")
    train(epoch)
    print("Testing")
    #test1(epoch)
    test(epoch)
    scheduler.step()
