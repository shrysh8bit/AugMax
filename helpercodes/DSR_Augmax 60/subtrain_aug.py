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
# epbatch = [0, 1160, 59, 868, 62, 57, 35, 1007, 80, 14, 31, 1182, 75, 77, 55, 78, 73, 636, 18, 1058, 846, 234, 390, 23, 283, 479, 26, 260, 585, 328, 30, 513, 491, 304, 34, 952, 66, 37, 38, 369, 225, 889, 42, 728, 44, 45, 313, 572, 793, 563, 426, 51, 52, 53, 54, 695, 540, 719, 58, 244, 614, 61, 566, 420, 64, 969, 459, 67, 1048, 69, 747, 71, 573, 233, 74, 394, 76, 81, 33, 79, 1091, 243, 3, 594, 84, 886, 1097, 537, 88, 543, 521, 91, 92, 481, 506, 938, 96, 97, 536, 990, 100, 635, 102, 685, 99, 463, 165, 1198, 232, 715, 110, 546, 386, 994, 1047, 115, 24, 512, 1166, 292, 580, 206, 673, 123, 124, 125, 379, 316, 436, 129, 120, 149, 132, 352, 441, 309, 533, 1046, 407, 139, 140, 1090, 1083, 318, 382, 398, 288, 147, 391, 487, 1113, 529, 511, 456, 5, 488, 781, 1022, 111, 159, 577, 720, 1021, 1015, 943, 25, 501, 1177, 878, 425, 358, 1037, 305, 104, 1, 490, 176, 1170, 1051, 11, 987, 1061, 303, 183, 383, 646, 743, 637, 995, 1169, 1165, 2, 1054, 626, 971, 195, 1023, 1059, 340, 437, 362, 480, 444, 203, 204, 796, 1197, 1027, 881, 209, 765, 564, 778, 213, 214, 242, 957, 1116, 72, 907, 241, 535, 174, 254, 1142, 271, 991, 547, 438, 229, 1033, 1063, 1109, 968, 311, 582, 1084, 852, 1167, 239]
#epbatch = [0, 483, 299, 392, 390, 1, 496, 7, 329, 384, 8, 2, 3, 4, 5, 346, 481, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 123, 129, 30, 31, 32, 171, 34, 35, 36, 37, 38, 39, 488, 41, 42, 43, 146, 168, 46, 47, 48, 49, 50, 9, 52, 125, 435, 55, 56, 57, 58, 59, 420, 61, 147, 63, 484, 472, 66, 67, 68, 69, 364, 71, 126, 290, 74, 75, 76, 77, 134, 79, 80, 81, 82, 466, 84, 85, 86, 262, 88, 89, 90, 155, 122, 93, 152, 151, 418, 97, 98, 130, 100, 142, 195, 471, 393, 105, 106, 107, 108, 6, 480, 438, 469, 120, 114, 485, 141, 487, 462, 128]

epbatch = [0, 661, 498, 3, 4, 5, 6, 7, 968, 9, 10, 954, 12, 648, 674, 657, 672, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 951, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 947, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 596, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 949, 119, 120, 121, 122, 616, 124, 125, 941, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 494, 964, 146, 147, 148, 149, 150, 151, 152, 945, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 938, 168, 169, 601, 171, 172, 173, 174, 175, 598, 177, 178, 179, 180, 181, 182, 962, 184, 185, 186, 187, 610, 189, 190, 191, 605, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 634, 209, 210, 211, 212, 213, 214, 215, 216, 586, 218, 219, 220, 221, 222, 223, 224, 934, 226, 227, 228, 229, 230, 231, 948, 233, 234, 235, 236, 944, 238, 239, 240, 241, 242, 243, 244, 933, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 931, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 145, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 486, 294, 295, 296, 624, 298, 299, 300, 301, 302, 303, 304, 930, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 647, 317, 318, 319, 320, 321, 322, 323, 324, 325, 587, 327, 328, 926, 330, 331, 594, 925, 334, 335, 921, 665, 476, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 571, 351, 576, 353, 589, 355, 356, 357, 358, 359, 360, 908, 362, 363, 364, 909, 366, 367, 895, 369, 370, 371, 912, 936, 374, 375, 376, 377, 378, 379, 380, 381, 382, 893, 887, 564, 329, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 466, 907, 402, 403, 404, 878, 406, 407, 873, 422, 923, 411, 885, 413, 414, 415, 416, 417, 11, 419, 420, 421, 877, 423, 424, 425, 426, 427, 428, 429, 890, 98, 432, 872, 874, 435, 552, 876, 871, 439, 440, 237, 870, 969, 444, 445, 446, 447, 857, 449, 619, 451, 452, 453, 454, 848, 852, 846, 844, 570, 460, 833, 462, 405, 464, 465, 841, 467, 468, 469, 840, 897, 535, 473, 474, 839, 232, 834, 829, 479, 480, 481, 579, 537, 126, 471, 825, 487, 488, 832, 824, 491, 492, 493, 816, 495, 879, 809, 811, 869, 500, 866, 502, 503, 915, 899, 813, 562, 563, 509, 510, 511, 512, 812, 514, 515, 516, 765, 918, 534, 649, 521, 522, 533, 524, 183, 526, 527, 528, 16, 350, 410, 799, 656, 798, 547, 536, 820, 782, 539, 780, 499, 956, 442, 544, 545, 546, 828, 548, 549, 855, 773, 935, 553, 794, 555, 556, 557, 851, 778, 560, 789, 781, 917, 779, 565, 1, 523, 772, 400, 561, 769, 572, 573, 574, 575, 606, 577, 753, 865, 580, 538, 582, 583, 584, 585, 167, 845, 588, 225, 590, 554, 592, 593, 929, 904, 801, 597, 144, 518, 600, 472, 352, 603, 823, 764, 758, 607, 608, 609, 599, 611, 612, 613, 641, 595, 520, 617, 757, 15, 620, 621, 622, 623, 808, 625, 626, 627, 384, 496, 946, 631, 632, 633, 507, 635, 636, 671, 752, 639, 8, 786, 642, 506, 644, 645, 646, 14, 504, 501, 650, 751, 13, 749, 654, 745, 744, 747, 658, 743, 660, 868, 640, 663, 664, 581, 666, 667, 668, 669, 748, 336, 118, 673, 333, 675, 676, 505, 519, 746, 383, 681, 682, 683, 684, 738, 530, 365, 688, 689, 740, 662, 731, 361, 900, 843, 372, 697, 698, 699, 700, 785, 756, 280, 704, 705, 706, 542, 807, 726, 724, 723, 558, 713, 714, 721, 720, 643, 718, 734]
 
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
        torch.save(state, './checkpoint_sub/ckpt_sub60_otheraug_all.pth')
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
