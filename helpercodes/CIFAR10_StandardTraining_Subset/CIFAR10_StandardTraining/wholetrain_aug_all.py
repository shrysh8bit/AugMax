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
import os
import copy
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
#trainind = list(random.sample(list(np.arange(0,60000)),6000))
test_transform = transforms.Compose([transforms.ToTensor()])
#trainind = np.load('./mnist_train_aug_indices.npy')
#print(trainind)
for batchid, (ip, targ) in enumerate(trainl):
    #if batchid in trainind:
    ftrain.append(ip.numpy())
    ytrain.append(targ.item())
    for _ in range(1):
        ftrain.append(np.expand_dims(aug(ip, test_transform, 3, 3).cpu().numpy(),axis=0))
        ytrain.append(targ.item())
    '''else:
        ftrain.append(ip.numpy())
        ytrain.append(targ.item())'''

############################ Adding Augmented Data ########################
trainset = AugmentedDataset(np.asarray(ftrain),np.asarray(ytrain))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)


print(len(trainloader))

testset = torchvision.datasets.MNIST(
    root='./data/', train=False, download=True, transform=transform)

############## Standard clean test set ###################
testloaderb = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# Model
torch.manual_seed(123)
print('==> Building model..')
net = Net()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
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
        torch.save(state, './checkpoint_sub/ckptw_other_aug_all.pth')
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
