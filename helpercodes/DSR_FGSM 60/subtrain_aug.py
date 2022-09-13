import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from subsetpack.fgsm import FGSM
import torchvision
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

net = Net()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# Data
print('==> Preparing data..')

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(
    root='./data/', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

for epoch in range(2):
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

atk = FGSM(net, eps=0.3)
trainl = torch.utils.data.DataLoader(trainset,shuffle=False,batch_size = 1)
ftrain = []
ytrain = []
random.seed(123)
trainind = list(random.sample(list(np.arange(0,60000)),6000))
#print(trainind)
#np.save('mnist_train_aug_indices.npy',trainind)

for batchid, (ip, targ) in enumerate(trainl):
    if batchid in trainind:
        ftrain.append(ip.numpy())
        ftrain.append(atk(ip,targ)[1].cpu().numpy())
        ytrain.append(targ.item())
        ytrain.append(targ.item())
    else:
        ftrain.append(ip.numpy())
        ytrain.append(targ.item())

trainset = AugmentedDataset(np.asarray(ftrain),np.asarray(ytrain))
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

#epbatch = [0, 262, 54, 3, 257, 256, 277, 219, 204, 37, 259, 26, 195, 258, 272, 224, 269, 205, 190, 19, 20, 21, 139, 14, 24, 226, 4, 192, 223, 29, 193, 183, 32, 33, 78, 57, 36, 182, 177, 228, 149, 64, 86, 170, 163, 45, 46, 47, 171, 180, 156, 239, 263, 53, 210, 30, 56, 89, 42, 59, 157, 174, 62, 261, 267, 22, 166, 67, 118, 148, 146, 15, 72, 100, 107, 265, 76, 209, 116, 132, 80, 81, 137, 83, 238, 141, 151, 87, 88, 188, 90, 91, 212, 12, 10, 95, 34, 203, 140, 99, 23, 134, 11, 220, 38, 9, 8, 131, 130, 2, 124, 111, 225, 122, 84, 216, 129, 120, 1, 147]
#epbatch = [0, 350, 200, 230, 102, 266, 202, 10, 161, 208, 257, 105, 228, 47, 69, 348, 188, 42, 6, 244, 20, 21, 195, 194, 24, 186, 214, 231, 73, 223, 33, 183, 32, 78, 13, 178, 36, 37, 330, 11, 224, 41, 34, 304, 30, 162, 321, 273, 177, 203, 245, 51, 104, 40, 59, 94, 56, 57, 58, 3, 83, 199, 100, 215, 5, 218, 98, 211, 68, 157, 70, 25, 72, 235, 167, 156, 76, 184, 31, 79, 80, 81, 166, 151, 1, 116, 86, 87, 88, 147, 90, 91, 92, 272, 44, 95, 96, 121, 238, 99, 217, 19, 190, 103, 311, 201, 52, 107, 108, 323, 110, 71, 15, 113, 220, 115, 175, 84, 118, 97, 136, 4, 122, 123, 124, 148, 133, 146, 2, 129, 130, 164]
#epbatch = [0, 359, 357, 264, 331, 339, 313, 315, 360, 314, 319, 261, 318, 334, 293, 312, 356, 304, 417, 192, 1, 148, 217, 195, 24, 249, 343, 178, 219, 29, 30, 302, 182, 354, 278, 183, 36, 37, 238, 265, 233, 341, 166, 235, 244, 204, 210, 243, 174, 586, 153, 338, 283, 241, 188, 443, 263, 180, 131, 289, 239, 593, 118, 193, 224, 303, 97, 228, 115, 136, 273, 281, 547, 223, 168, 351, 659, 329, 639, 277, 626, 269, 122, 187, 266, 256, 512, 203, 99, 96, 95, 297, 92, 50, 120, 342, 91, 87, 86, 84, 119, 19, 104, 81, 80, 79, 72, 275, 327, 123, 62, 59, 102, 246, 134, 88, 144, 52, 70, 285, 55, 284, 11, 10, 73, 348, 164, 163, 141, 116, 577, 157]
#Selected with eps=0.3 train, eps = 0.3 validation
#epbatch = [0, 191, 266, 94, 47, 157, 187, 189, 263, 122, 196, 203, 258, 204, 164, 188, 210, 136, 86, 78, 20, 235, 195, 184, 24, 81, 63, 183, 182, 17, 30, 177, 32, 19, 166, 113, 36, 48, 206, 172, 243, 111, 42, 43, 44, 313, 208, 87, 31, 170, 237, 257, 171, 53, 54, 192, 56, 57, 269, 59, 46, 131, 62, 247, 41, 156, 2, 91, 68, 69, 37, 246, 72, 73, 74, 151, 76, 77, 11, 79, 80, 242, 213, 83, 84, 85, 211, 167, 214, 312, 26, 328, 3, 148, 224, 95, 96, 307, 70, 99, 265, 146, 141, 103, 147, 217, 233, 140, 223, 199, 202, 239, 112, 6, 114, 115, 194, 135, 10, 305, 118, 121, 7, 123, 124, 125, 1, 267, 133, 105, 130, 4]
#Selected with eps=0.3 train, eps = 0.1 validation
#epbatch = [0, 1, 44, 252, 4, 180, 6, 7, 235, 244, 10, 150, 121, 185, 190, 204, 188, 84, 194, 186, 178, 21, 169, 196, 24, 155, 198, 189, 124, 29, 151, 31, 32, 33, 34, 183, 36, 37, 215, 170, 77, 174, 42, 159, 141, 45, 46, 47, 85, 79, 182, 166, 193, 53, 54, 19, 56, 57, 171, 59, 167, 63, 102, 184, 164, 97, 210, 86, 68, 69, 70, 143, 72, 73, 219, 156, 76, 212, 41, 202, 80, 81, 147, 100, 208, 145, 161, 87, 88, 89, 90, 91, 67, 93, 165, 2, 96, 15, 195, 99, 83, 148, 92, 115, 140, 217, 106, 107, 163, 71, 149, 111, 112, 20, 137, 123, 116, 146, 118, 119, 11, 142, 122, 135, 61, 134, 177, 136, 133, 129, 130, 131]
#Selected with eps=0.3 train, eps = 0.5 validation
# epbatch = [0, 311, 220, 266, 4, 73, 238, 306, 258, 11, 303, 277, 114, 13, 5, 282, 267, 160, 249, 46, 20, 181, 233, 263, 24, 183, 228, 36, 182, 250, 25, 31, 32, 172, 1, 205, 313, 10, 170, 166, 171, 41, 42, 224, 164, 45, 251, 281, 15, 167, 50, 51, 142, 53, 54, 156, 56, 57, 174, 122, 14, 3, 278, 107, 153, 265, 242, 67, 219, 105, 124, 213, 168, 18, 74, 150, 76, 22, 308, 79, 80, 81, 152, 235, 84, 257, 190, 87, 88, 112, 62, 204, 193, 151, 78, 269, 96, 283, 178, 99, 146, 210, 102, 103, 179, 260, 261, 212, 141, 95, 110, 111, 137, 35, 239, 115, 116, 312, 118, 148, 58, 19, 223, 123, 203, 134, 8, 133, 135, 2, 192, 254]
# FGSM 60 indices
# epbatch = [0, 595, 640, 3, 4, 5, 6, 7, 8, 627, 10, 654, 539, 175, 597, 15, 586, 639, 206, 583, 169, 537, 22, 23, 24, 594, 26, 579, 578, 577, 215, 205, 32, 604, 34, 569, 36, 37, 522, 636, 560, 41, 42, 635, 619, 45, 634, 47, 633, 620, 558, 466, 52, 53, 54, 55, 56, 57, 574, 575, 553, 547, 62, 63, 326, 533, 464, 67, 68, 608, 70, 626, 72, 476, 617, 552, 76, 622, 591, 127, 80, 81, 82, 83, 84, 85, 280, 87, 557, 544, 90, 91, 152, 540, 373, 95, 96, 97, 520, 465, 613, 565, 646, 103, 39, 161, 106, 279, 551, 423, 590, 200, 538, 113, 114, 541, 116, 451, 118, 119, 120, 330, 122, 123, 124, 108, 536, 112, 444, 428, 621, 131, 254, 65, 134, 135, 469, 137, 517, 139, 140, 624, 196, 143, 138, 145, 146, 93, 148, 149, 259, 151, 159, 153, 637, 155, 629, 618, 570, 615, 160, 515, 162, 163, 64, 165, 166, 641, 652, 518, 170, 171, 512, 413, 244, 335, 337, 177, 178, 49, 412, 509, 507, 183, 599, 426, 427, 187, 172, 189, 190, 191, 192, 193, 194, 195, 593, 505, 511, 415, 638, 644, 202, 203, 204, 429, 485, 649, 271, 651, 462, 519, 212, 213, 405, 399, 493, 217, 494, 219, 220, 477, 473, 223, 224, 225, 499, 645, 487, 362, 210, 453, 408, 358, 378, 235, 104, 481, 238, 239, 480, 250, 242, 377, 632, 74, 475, 273, 248, 249, 659, 379, 658, 606, 631, 77, 256, 261, 258, 348, 503, 472, 470, 460, 343, 265, 266, 267, 467, 269, 406, 647, 272, 314, 630, 461, 581, 277, 278, 446, 247, 281, 282, 306, 59, 285, 286, 648, 361, 614, 642, 422, 458, 141, 226, 336, 456, 452, 628, 253, 174, 529, 508, 58, 448, 592, 492, 307, 92, 445, 397, 311, 312, 313, 395, 315, 305, 317, 318, 442, 526, 228, 483, 1, 150, 167, 567, 437, 328, 329, 420, 523, 374, 185, 334, 545, 107, 341, 561, 339, 370, 532, 342, 371, 299, 435, 304, 164, 424, 35, 419, 351, 352, 2, 354, 491, 11, 357, 596, 434, 360, 338, 623, 156, 31, 410, 366, 367, 436, 564, 20, 275, 372, 457, 99, 375, 376, 404, 19, 214, 549, 381, 322, 463, 102, 655, 403, 387, 398, 389, 603, 391, 392, 657, 61, 332]
epbatch = [0, 661, 498, 3, 4, 5, 6, 7, 968, 9, 10, 954, 12, 648, 674, 657, 672, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 951, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 947, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 596, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 949, 119, 120, 121, 122, 616, 124, 125, 941, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 494, 964, 146, 147, 148, 149, 150, 151, 152, 945, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 938, 168, 169, 601, 171, 172, 173, 174, 175, 598, 177, 178, 179, 180, 181, 182, 962, 184, 185, 186, 187, 610, 189, 190, 191, 605, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 634, 209, 210, 211, 212, 213, 214, 215, 216, 586, 218, 219, 220, 221, 222, 223, 224, 934, 226, 227, 228, 229, 230, 231, 948, 233, 234, 235, 236, 944, 238, 239, 240, 241, 242, 243, 244, 933, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 931, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 145, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 486, 294, 295, 296, 624, 298, 299, 300, 301, 302, 303, 304, 930, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 647, 317, 318, 319, 320, 321, 322, 323, 324, 325, 587, 327, 328, 926, 330, 331, 594, 925, 334, 335, 921, 665, 476, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 571, 351, 576, 353, 589, 355, 356, 357, 358, 359, 360, 908, 362, 363, 364, 909, 366, 367, 895, 369, 370, 371, 912, 936, 374, 375, 376, 377, 378, 379, 380, 381, 382, 893, 887, 564, 329, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 466, 907, 402, 403, 404, 878, 406, 407, 873, 422, 923, 411, 885, 413, 414, 415, 416, 417, 11, 419, 420, 421, 877, 423, 424, 425, 426, 427, 428, 429, 890, 98, 432, 872, 874, 435, 552, 876, 871, 439, 440, 237, 870, 969, 444, 445, 446, 447, 857, 449, 619, 451, 452, 453, 454, 848, 852, 846, 844, 570, 460, 833, 462, 405, 464, 465, 841, 467, 468, 469, 840, 897, 535, 473, 474, 839, 232, 834, 829, 479, 480, 481, 579, 537, 126, 471, 825, 487, 488, 832, 824, 491, 492, 493, 816, 495, 879, 809, 811, 869, 500, 866, 502, 503, 915, 899, 813, 562, 563, 509, 510, 511, 512, 812, 514, 515, 516, 765, 918, 534, 649, 521, 522, 533, 524, 183, 526, 527, 528, 16, 350, 410, 799, 656, 798, 547, 536, 820, 782, 539, 780, 499, 956, 442, 544, 545, 546, 828, 548, 549, 855, 773, 935, 553, 794, 555, 556, 557, 851, 778, 560, 789, 781, 917, 779, 565, 1, 523, 772, 400, 561, 769, 572, 573, 574, 575, 606, 577, 753, 865, 580, 538, 582, 583, 584, 585, 167, 845, 588, 225, 590, 554, 592, 593, 929, 904, 801, 597, 144, 518, 600, 472, 352, 603, 823, 764, 758, 607, 608, 609, 599, 611, 612, 613, 641, 595, 520, 617, 757, 15, 620, 621, 622, 623, 808, 625, 626, 627, 384, 496, 946, 631, 632, 633, 507, 635, 636, 671, 752, 639, 8, 786, 642, 506, 644, 645, 646, 14, 504, 501, 650, 751, 13, 749, 654, 745, 744, 747, 658, 743, 660, 868, 640, 663, 664, 581, 666, 667, 668, 669, 748, 336, 118, 673, 333, 675, 676, 505, 519, 746, 383, 681, 682, 683, 684, 738, 530, 365, 688, 689, 740, 662, 731, 361, 900, 843, 372, 697, 698, 699, 700, 785, 756, 280, 704, 705, 706, 542, 807, 726, 724, 723, 558, 713, 714, 721, 720, 643, 718, 734]

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

testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)

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
        torch.save(state, './checkpoint_sub/ckpt_eps3_f60_aug.pth')
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
