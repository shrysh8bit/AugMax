import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from scipy.linalg import lstsq
from sklearn.preprocessing import normalize
from subsetpack.carlini_wagner import carlini_wagner_l2
from subsetpack.fgsm import FGSM
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from subsetpack.augmentations import augmentations
import numpy as np
import pickle
import random
import os
import argparse
import time
import copy
from numpy import linalg as LA

class HelperFunc(object):

	def __init__(self,train,test,model,confdata):

		self.num_entry = 0
		self.train_loader=train
		self.test_loader=test
		self.model = model
		self.dv2o = np.zeros((len(self.test_loader),1))	
		self.cv_val = np.zeros((len(self.test_loader),len(self.train_loader)))		
		self.rootpath = confdata['root_dir']
		self.resume = confdata['resume']
		self.csel = confdata['csel']
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.selected = [[] for i in range(confdata['epochs'])]
		self.numtp = confdata['num_trajpoint']
		self.csel_epoch_interval = confdata['num_freqep']
		self.csel_batch_interval = confdata['num_freqbatch']
		#self.batchl = [i for i in range(0,len(self.train_loader),self.csel_batch_interval)]
		self.batchcount = len(self.train_loader)//self.csel_batch_interval if len(self.train_loader)%2!=0 else (len(self.train_loader)//self.csel_batch_interval)- 1
		self.lastepoch = list(np.arange(0,confdata['epochs'],confdata['num_freqbatch']))[-1]
		self.criterion = nn.CrossEntropyLoss(reduction='mean')
		self.criterion_indiv = nn.CrossEntropyLoss(reduction='none')
		self.lastbatch = len(self.train_loader)-1
		self.softprob = nn.Softmax(dim=1)
		self.net1 = self.model
		self.model.to(self.device)
		self.net1.to(self.device)
		if not os.path.exists(self.rootpath):
			os.mkdir(self.rootpath)
		if not os.path.exists(self.rootpath+'trajp/'):
			os.mkdir(self.rootpath+'trajp/')
		if not os.path.exists(self.rootpath+'misc/'):
			os.mkdir(self.rootpath+'misc/')

		self.lr = 0.1
		self.So = []
		self.Soe = []
		self.reslisto = []
		self.Io = []
		self.Eo = []
		self.dvno = []
		self.esno = []

	def calc_loss(self,y, t):
		#criterion = nn.CrossEntropyLoss()
		loss = self.criterion(y,t)
		return loss

	def calc_loss_test(self,y,y_a, t):
		#criterion = nn.CrossEntropyLoss()
		loss = 0.5*self.criterion(y,t)+0.5*self.criterion(y_a,t)
		return loss

	def get_grad(self,input, target, model):

		model.eval()
		z, t = input.to(self.device), target.to(self.device)
		y = model(z)
		loss = self.calc_loss(y, t)
		# Compute sum of gradients from model parameters to loss
		params = model.fc2.weight
		result = list(grad(loss, params))[0].detach()

		return result

	def get_grad_test(self,loss, model):

		model.eval()
		# Compute sum of gradients from model parameters to loss
		params = model.fc2.weight
		result = list(grad(loss, params))[0].detach()
		del loss
		return result

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
		image_aug = copy.deepcopy(image)
		transform = transforms.ToPILImage()
		image_aug = transform(image_aug)
		depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
		for _ in range(depth):
		    op = np.random.choice(aug_list)
		    image_aug = op(image_aug, aug_severity)
		
		image_aug = preprocess(image_aug)

		return image_aug


	# Value function callback that computes value function delta and aggregate estimate C
	def valuefunc_cb(self,epoch,batch_idx,inputs,targets,lossval,cz,czs,initmodel,model):

		rbloss=[0 for i in range(len(self.test_loader))]
		#atk = FGSM(model, eps=0.3)
		param = copy.deepcopy(model)	
		if batch_idx==0: #first time computing value function
			if os.path.exists(self.rootpath+'misc/lastretain.pth'):
				self.net1 = torch.load(self.rootpath+'misc/lastretain.pth')
				if self.resume and self.num_entry==0:
					self.batchgrad = self.get_grad(inputs, targets, self.net1)
			else:
				self.net1 = copy.deepcopy(initmodel)
				self.batchgrad = self.get_grad(inputs, targets, initmodel)
		#else:
		model.eval()
		self.net1.eval()
		#atk_curr = FGSM(model, eps=0.3)
		#atk_prev = FGSM(self.net1, eps=0.3)
		test_transform = transforms.Compose([transforms.ToTensor()])
		
		for testi, data in enumerate(self.test_loader,0):
			delv = []
			delvc = []
			adv_images = []
			inputs_test, targets_test= data
			inputs_test, targets_test = inputs_test.to(self.device), targets_test.to(self.device)
			#lv_c,adv_images = atk_curr(inputs_test, targets_test)
			for index in range(len(inputs_test)):
				adv_images.append(self.aug(inputs_test[index], test_transform, -1, 3))
			adv_images = torch.stack(adv_images).to(self.device)
			#print(inputs_test.shape)
			#print(adv_images.shape)
			predn = model(inputs_test)
			predn_a = model(adv_images)

			#lv_o,adv_inputs_o = atk_prev(inputs_test, targets_test)
			predn1 = self.net1(inputs_test)
			#predn1_a = self.net1(adv_inputs_o)
			predn1_a = self.net1(adv_images)

			self.batchgrad = self.get_grad(inputs, targets, self.net1)
			lossn = self.calc_loss_test(predn, predn_a, targets_test)
			lossn1 = self.calc_loss_test(predn1, predn1_a, targets_test)
			testgrad = self.get_grad_test(lossn1,self.net1)
			#lossval[testi] = lossval[testi]+(torch.mean(lossn).item()-torch.mean(lossn1).item())
			lossval[testi] = lossval[testi]+(lossn.item()-lossn1.item())
			rbloss[testi] = lossn.item()-lossn1.item()
			#rbloss[testi] = torch.mean(lossn).item()-torch.mean(lossn1).item()
			#\nabla loss_test \nabla loss_batch
			cmat = torch.sum(testgrad*self.batchgrad).item()
			#Second order approximation
			czs[testi].append(cmat+0.5*(cmat*cmat))
			cz[testi].append(cmat)

		model.train()              
		self.net1 = copy.deepcopy(model)
			
		if batch_idx==self.lastbatch: #last batch when checksel is run
			param.load_state_dict(self.net1.state_dict())
			torch.save(param,self.rootpath+'misc/lastretain.pth')

		return lossval,cz,czs,rbloss


	#ALgorithm 1 for saving selected trajectory points along with importance weights/values
	#Calls Algorithm 2: CheckReplace for substituting the already selected trajectory points with better trajectories giving better approximation
	def checksel_cb(self,dv2w,cvs2w,model,epoch):
		# Number of to-be selected checkpoints

		if self.csel is True and epoch%self.csel_epoch_interval==0:
			ko = self.numtp
		
			paramc = copy.deepcopy(model)
			dv2w = np.array(dv2w)
			cvs2w = np.array(cvs2w)
			#print(cvs2w.shape)
			if self.resume and self.num_entry==0:
				alphao = np.load(self.rootpath+'misc/alpha_val.npy')
				esto = np.load(self.rootpath+'misc/estimate.npy')
				self.dv2o = np.load(self.rootpath+'misc/valuefunc.npy')
				self.cv_val = np.load(self.rootpath+'misc/est_valuefunc.npy')
				'''Soval = np.load(self.rootpath+'misc/countind_val.npy')
				for elem in range(Soval.shape[0]):
					self.So.append((Soval[elem][0],Soval[elem][1]))'''
				Soval = np.load(self.rootpath+'misc/cpind_val.npy')
				for elem in range(Soval.shape[0]):
					self.So.append((Soval[elem][0],Soval[elem][1])) 
				self.Sco = np.load(self.rootpath+'misc/estimate_grad.npy')

			if epoch>0 and not(self.resume and self.num_entry==0):
				alphao = lstsq(self.Sco.dot(self.Sco.T),self.Sco.dot(self.dv2o))[0] #update alpha (line 11)
				esto = np.dot(self.Sco.T,alphao.reshape(-1,1)) #update eta (line 12)    

			self.dv2o = self.dv2o + dv2w.reshape(-1,1) #I : dv2o
			self.cv_val = self.cv_val + cvs2w
			#print("Shape of cvs")
			#print(cvs2w.shape)
			cv2o = normalize(self.cv_val, axis = 0) #C: normalise (line 9)
			
			#print(cv2o.shape)
			print("Epoch "+str(epoch))
			print("Selections count "+str(len(self.So)))
			for batch in range(cv2o.shape[1]):
				if len(self.So) < ko:
					self.So.append((epoch,batch)) 
					#self.Soe.append((epoch,self.batchl[batch]))
					if batch == 0:
						self.Sco = -self.lr*(cv2o[:,batch]) # Adding to S (line 19)
					else:
						self.Sco = np.vstack([self.Sco,-self.lr*(cv2o[:,batch])]) # Adding to S (line 19)
						alphao = lstsq(self.Sco.dot(self.Sco.T),self.Sco.dot(self.dv2o))[0] # Update alpha: line 20
						esto = np.dot(self.Sco.T,alphao.reshape(-1,1)) #Update eta: line 20
				else:
					####### CheckReplace module ######
					#print(self.Soe)
					alphao,esto,self.Sco,reso,self.So= self.checkreplace(self.dv2o,esto,cv2o,alphao,epoch,batch)
					#print("Length So")
					#print(len(self.So))
					
			self.dvno.append(LA.norm(self.dv2o))
			self.esno.append(LA.norm(esto))
			print("Selections(Epoch, batch)")
			print(self.So) #(epoch,batch)
			#self.selco[int(epoch-self.start_epoch/self.freqep)].append(self.Soe)
			self.reslisto.append(LA.norm(reso)/LA.norm(self.dv2o))
			print("Residuals")
			print(self.reslisto)
			self.Io.append(self.dv2o)
			self.Eo.append(esto)

			for sl in range(len(self.So)):
				ep,bt = self.So[sl]
				if ep == epoch:
					self.selected[ep].append(bt)

			np.save(self.rootpath+'misc/alpha_val.npy',alphao)
			np.save(self.rootpath+'misc/valuefunc.npy',np.asarray(self.dv2o))
			np.save(self.rootpath+'misc/est_valuefunc.npy',np.asarray(self.cv_val))
			np.save(self.rootpath+'misc/estimate.npy',np.asarray(esto))
			np.save(self.rootpath+'misc/estimate_grad.npy',np.asarray(self.Sco))
			np.save(self.rootpath+'misc/cpind_val.npy',self.So)
			#np.save(self.rootpath+'misc/countind_val.npy',self.So)
			np.save(self.rootpath+'misc/cumulloss_val.npy',np.asarray(self.Io))
			np.save(self.rootpath+'misc/estimate_val.npy',np.asarray(self.Eo))
			self.num_entry = self.num_entry + 1
			cpv = np.load(self.rootpath+'misc/cpind_val.npy')
			val_ind = []
			for elem in range(cpv.shape[0]):
				val_ind.append((alphao[elem][0],cpv[elem][0],cpv[elem][1]))
			np.save(self.rootpath+'trajp_value_indices.npy',np.asarray(val_ind))
			print(epoch)
			print(self.selected[epoch])

		else:
			pass


	#Algorithm 2 replaces existing trajectory points with new ones that better approximate the value function
	def checkreplace(self,dv2o,esto,cv2o,alphao,epoch,batch):

		reso = dv2o - esto #residual vector : line 1
		#print(dv2o.shape)
		#print(esto.shape)
		maxproj = -np.inf
		ind = None
		exbatch = []
		#print(cv2o.shape)
		for s in range(self.Sco.shape[0]):
			self.Sco[s]=-self.lr*cv2o[:,self.So[s][1]]
			self.So[s]= (epoch,self.So[s][1])
			exbatch.append(self.So[s][1])

		alphao = lstsq(self.Sco.dot(self.Sco.T),self.Sco.dot(dv2o))[0] # Update alpha : line 15
		esto = np.dot(self.Sco.T,alphao.reshape(-1,1)) # Update eta : line 15

		if batch not in exbatch:
			for s in range(self.Sco.shape[0]):
				nvo = reso + (alphao[s]*(self.Sco[s].reshape(-1,1))) #new residual on removing j : line 5
				#print(nvo.shape)
				nprojo = np.dot(-self.lr*cv2o[:,batch].reshape(-1,1).T,nvo) #proj of C_i on new residual : line 6
				eprojo = np.dot(self.Sco[s].reshape(-1,1).T,nvo) #proj of C_j on new residual : line 7
				#lines 8 - 10
				if nprojo > eprojo and nprojo > maxproj:
					projdiff = nprojo - eprojo
					maxproj = eprojo
					ind = s

			#lines 13 - 14
			if ind!=None:
				'''for selected in self.So:
					if selected[1]!=batch:
						addelem = 1
					else:
						addelem = 0
						break'''
				#if addelem!=0:		
				self.Sco[ind] = -self.lr*cv2o[:,batch]
				self.So[ind] = (epoch,batch)
				#self.Soe[ind] = (epoch,self.batchl[batch])

			alphao = lstsq(self.Sco.dot(self.Sco.T),self.Sco.dot(dv2o))[0] # Update alpha : line 15
			esto = np.dot(self.Sco.T,alphao.reshape(-1,1)) # Update eta : line 15

		return alphao,esto,self.Sco,reso,self.So
