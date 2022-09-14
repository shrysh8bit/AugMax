import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from scipy.linalg import lstsq
from sklearn.preprocessing import normalize
from torchvision import datasets
from torch.autograd import grad
from subsetpack.dataset import RandomBatchSampler
from subsetpack.model import Model
from subsetpack.cw import CW
from subsetpack.pgd import PGD
from subsetpack.fgsm import FGSM
from subsetpack.helper import HelperFunc
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import random
import os
import argparse
from subsetpack.augmentations import augmentations
import time
import copy
from numpy import linalg as LA
import scipy.stats
from subsetpack.helper import HelperFunc
from PIL import Image
class AugmentedDataset(torch.utils.data.Dataset):

  def __init__(self, X, y, scale_data=True):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)
      #print("inside")
      print(self.X.shape)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]

class TrajSel(object):

	def __init__(self,trainset,train,test,model,helpobj,confdata):

		self.trainset = trainset
		self.train_loader=train
		self.test_loader=test
		self.model = model
		self.rootpath = confdata['root_dir']
		self.resume = confdata['resume']
		self.epochs = confdata['epochs']
		self.numtp = confdata['num_trajpoint']
		self.unif_epoch_interval = self.epochs//self.numtp
		self.numfp = confdata['num_freqcp']
		self.csel_epoch_interval = confdata['num_freqep']
		self.csel_batch_interval = confdata['num_freqbatch']
		self.batchl = [i for i in range(0,len(self.train_loader),self.csel_batch_interval)]
		self.step_in_epoch = 0
		self.csel = confdata['csel']
		self.helpobj = helpobj
		self.confdata = confdata
		self.count = 0
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.criterion = nn.CrossEntropyLoss()
		if self.resume:
			checkpointpath = torch.load(self.rootpath+'checkpoint/ckpt.pth')
			self.model.load_state_dict(checkpointpath['model'])
			self.start_epoch = checkpointpath['epoch']+1
		else:
			self.start_epoch = 0
		self.lr = 0.1
		self.optimizer = optim.SGD(self.model.parameters(), self.lr)
		#self.model.load_state_dict(torch.load('./checkpoint_sub/ckpt.pth')['net'])
		self.net1 = self.model
		self.initmodel = self.model
		
		self.model.to(self.device)
		self.net1.to(self.device)
		if not os.path.exists(self.rootpath):
			os.mkdir(self.rootpath)
		if not os.path.exists(self.rootpath+'checkpoint/'):
			os.mkdir(self.rootpath+'checkpoint/')
		if not(self.resume) and os.path.exists(self.rootpath+'misc/lastretain.pth'):
			os.remove(self.rootpath+'misc/lastretain.pth')



	def initialize(self):

		lossval = [0 for i in range(len(self.test_loader))]
		cz = [[] for i in range(len(self.test_loader))]
		czs = [[] for i in range(len(self.test_loader))]

		return lossval,cz,czs

	def savemodel(self,netv=None,batchid=None,epoch=None,unif=False):

		if netv!=None:
			netv[batchid] = self.model.state_dict()
			self.count = self.count + 1

		if self.step_in_epoch!=0 and self.step_in_epoch%self.numfp==0 or self.step_in_epoch==len(self.train_loader)-1:
			state = {'model': self.model.state_dict(),'epoch': epoch}
			torch.save(state, self.rootpath+'checkpoint/ckpt.pth')
		if unif==True:
			torch.save(self.model.state_dict(), self.rootpath+'checkpoint/epoch_'+str(epoch)+'.pth')


	def aug(self,image, preprocess, mixture_depth, aug_severity):
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

	########## Trains the model on a dataset; runs CheckSel at an interval epoch; stores the miscellaneous results and also the trajectory indices with their weights ###########
	def fit(self):

		#print(self.device)
		diffrbloss = []
		batch_rbloss_change = []
		total_rbloss_change = []
		corrlist = []
		dotepochs = []
		trainl = torch.utils.data.DataLoader(self.trainset,shuffle=False,batch_size = 1)
		ftrain = []
		ytrain = []
		#random.seed(123)
		#trainind = list(random.sample(list(np.arange(0,60000)),6000))
		trainind = np.load('./mnist_train_aug_indices.npy')
		test_transform = transforms.Compose([transforms.ToTensor()])
		#print(trainind)
		for batchid, (ip, targ) in enumerate(trainl):
			if batchid in trainind:
				#print(ip.numpy().shape)
				ftrain.append(ip.numpy())
				ytrain.append(targ.item())
				for _ in range(1):
				    ftrain.append(np.expand_dims(self.aug(ip, test_transform, 3, 3).cpu().numpy(),axis=0))
				    #ftrain.append(atk(ip,targ)[1].cpu().numpy())
				    ytrain.append(targ.item())
				#print(ftrain[0])
				#print(ftrain[1])
				#exit(1)
			else:
				ftrain.append(ip.numpy())
				ytrain.append(targ.item())

		self.trainset = AugmentedDataset(np.asarray(ftrain),np.asarray(ytrain))
		sampler = RandomBatchSampler(torch.utils.data.sampler.BatchSampler(torch.utils.data.sampler.SequentialSampler(self.trainset),batch_size=self.confdata['trainbatch'],drop_last=False))
		self.trainloader = torch.utils.data.DataLoader(self.trainset,batch_sampler=sampler)
		print(len(self.trainloader))
		self.helpobj = HelperFunc(self.trainloader,self.test_loader,self.model,self.confdata)
		for epoch in range(self.start_epoch,self.start_epoch+self.epochs):

			eptime = time.time()
			self.count = 0
			trloss = 0
			corrval = []
			batch_rbloss_change = []
			lossval, cz, czs = self.initialize()
			self.model.train()

			print("Epoch")
			print(epoch)
			
			for batch_idx, (inputs, targets) in enumerate(self.trainloader):

				start = time.time()
				#print("Batch idx")
				#print(batch_idx)
				self.model.train()
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				inputs = inputs.squeeze(1)
				#print(adv_images.shape)			
				if self.csel is True and epoch%self.csel_epoch_interval==0:
					# Value function callback returning value function delta_{ind} and aggregate of estimate over all datapoints in the batch C_{ind}
					#lossval,cz,czs = self.helpobj.valuefunc_cb(epoch,batch_idx,inputs,targets,lossval,cz,czs,self.initmodel,self.model)	
					lossval,cz,czs,rbloss = self.helpobj.valuefunc_cb(epoch,batch_idx,inputs.to(self.device),targets,lossval,cz,czs,self.initmodel,self.model)	
					#batch_rbloss_change.append(np.sum(rbloss,axis=0))
					#print(np.asarray(rbloss).reshape(-1,1).shape)
					#print((np.asarray(czs)[:,batch_idx]).reshape(-1,1).shape)
					#corrval.append(scipy.stats.pearsonr(np.asarray(rbloss),(np.asarray(czs)[:,batch_idx])))

				self.model.train()
				self.optimizer.zero_grad()
				outputs = self.model(inputs.to(self.device))
				loss = self.criterion(outputs, targets)
				#print("Loss")
				#print(loss)
				loss.backward()
				self.optimizer.step()

				self.savemodel(epoch=epoch)
				self.step_in_epoch+=1

				trloss = trloss + loss.item()
				#print("Batch time")
				#print(time.time()-start)
			# Saving uniformly spaced trajectories
			print("Epoch time")
			print(time.time()-eptime)
			'''if epoch%self.unif_epoch_interval==0:
				self.savemodel(epoch=epoch,unif=True)'''
			print('Epoch '+str(epoch)+', Loss '+str(trloss/len(self.train_loader)))			
			#checksel callback function to run Algorithm 1: CheckSel
			self.helpobj.checksel_cb(lossval,czs,self.model,epoch)