'''
Tiny-ImageNet:
Download by wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
Run python create_tin_val_folder.py to construct the validation set. 

Tiny-ImageNet-C:
Download by wget https://zenodo.org/record/2469796/files/TinyImageNet-C.tar?download=1
Run python dataloaders/fix_tin_c.py to remove the redundant images in TIN-C.

Tiny-ImageNet-V2:
Download ImageNet-V2 from http://imagenetv2public.s3-website-us-west-2.amazonaws.com/
Run python dataloaders/construct_tin_v2.py to select 200-classes from the full ImageNet-V2 dataset.

https://github.com/snu-mllab/PuzzleMix/blob/master/load_data.py
'''

import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np 
import os, shutil

def tiny_imagenet_dataloaders(data_dir, transform_train=True, AugMax=None, **AugMax_args):
    
    train_root = os.path.join(data_dir, 'train')  # this is path to training images folder
    validation_root = os.path.join(data_dir, 'val/images')  # this is path to validation images folder
    print('Tiny imagenet Training images loading from %s' % train_root)
    print('Tiny imagenet Validation images loading from %s' % validation_root)
    
    if AugMax is not None:
        preprocess = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        
        train_transform = transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])



        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                # transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                # transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'test': transforms.Compose([
                # transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        

        train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'),transform=data_transforms['train'])
        test_data = datasets.ImageFolder(os.path.join(data_dir, 'val'),transform=data_transforms['val']) 


        train_data = AugMax(train_data, data_transforms['train'], 
            mixture_width=AugMax_args['mixture_width'], mixture_depth=AugMax_args['mixture_depth'], aug_severity=AugMax_args['aug_severity'], 
        )

    else:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor()]) if transform_train else transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])

        train_data = datasets.ImageFolder(train_root, transform=train_transform)
        test_data = datasets.ImageFolder(validation_root, transform=test_transform)
    
    return train_data, test_data

# def tiny_imagenet_dataloaders(data_dir, transform_train=True, AugMax=None, **AugMax_args):
    
#     train_root = os.path.join(data_dir, 'train')  # this is path to training images folder
#     validation_root = os.path.join(data_dir, 'val/images')  # this is path to validation images folder
#     print('Training images loading from %s' % train_root)
#     print('Validation images loading from %s' % validation_root)
    
#     if AugMax is not None:
#         train_transform = transforms.Compose(
#             [transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(64, padding=4)]) if transform_train else None
#         test_transform = transforms.Compose([transforms.ToTensor()])

#         train_data = datasets.ImageFolder(train_root, transform=train_transform)
#         test_data = datasets.ImageFolder(validation_root, transform=test_transform)
#         train_data = AugMax(train_data, test_transform, 
#             mixture_width=AugMax_args['mixture_width'], mixture_depth=AugMax_args['mixture_depth'], aug_severity=AugMax_args['aug_severity'], 
#         )

#     else:
#         train_transform = transforms.Compose(
#             [transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(64, padding=4),
#             transforms.ToTensor()]) if transform_train else transforms.Compose([transforms.ToTensor()])
#         test_transform = transforms.Compose([transforms.ToTensor()])

#         train_data = datasets.ImageFolder(train_root, transform=train_transform)
#         test_data = datasets.ImageFolder(validation_root, transform=test_transform)
    
#     return train_data, test_data


def tiny_imagenet_deepaug_dataloaders(data_dir, transform_train=True, AugMax=None, **AugMax_args):
    
    train_root = os.path.join(data_dir, 'train')  # this is path to training images folder
    print('Training images loading from %s' % train_root)
    
    if AugMax is not None:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4)]) if transform_train else None
        test_transform = transforms.Compose([transforms.ToTensor()])

        train_data = datasets.ImageFolder(train_root, transform=train_transform)
        train_data = AugMax(train_data, test_transform, 
            mixture_width=AugMax_args['mixture_width'], mixture_depth=AugMax_args['mixture_depth'], aug_severity=AugMax_args['aug_severity'], 
        )

    else:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor()]) if transform_train else transforms.Compose([transforms.ToTensor()])

        train_data = datasets.ImageFolder(train_root, transform=train_transform)
    
    return train_data

def tiny_imagenet_c_testloader(data_dir, corruption, severity, 
    test_batch_size=1000, num_workers=4):

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_root = os.path.join(data_dir, corruption, str(severity))
    test_c_data = datasets.ImageFolder(test_root,transform=test_transform)
    test_c_loader = DataLoader(test_c_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_c_loader        

