######### This class will compute values/importance weights for all data points using selected trajectories from CheckSel #############

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import argparse
import time
import pickle


class DataValueCheckSel(object):

    def __init__(self,train,test,model,helperobj,confdata):

        self.trainset = train
        self.testloaderb = test
        self.model = model
        self.rootpath = confdata['root_dir']
        self.featdim = confdata['featuredim']
        self.bsize = confdata['trainbatch']
        self.keeprem = confdata['retain_scores']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resume = confdata['resume']
        self.fdim = confdata['featuredim']
        self.layer = self.model._modules.get('avgpool')
        checkpointpath = torch.load(self.rootpath+'checkpoint/ckpt.pth')
        self.resf = self.model
        self.resf.load_state_dict(checkpointpath['model'])
        self.resf.to(self.device)
        self.resf.eval()
        self.helperobj = helperobj

    ############# Finds the neighbour using computed features #############
    def findnearest(self,node, nodes):
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node)**2, axis=1)
        return np.argmin(dist_2)

    ############ Computes values for datapoints assigned to the selected trajectories during training ###########
    ############ For e.g. trajectory epoch10_batch15 will be associated with batch15 data where the batch id is already fixed during initial dataloading #########

    def get_feature(self,image):
        embed = torch.zeros(self.fdim)
        def copy_data(m,i,o):
            embed.copy_(o.data.reshape(o.data.size(1)))
        h = self.layer.register_forward_hook(copy_data)
        self.resf(image)
        h.remove()
        return embed

    def score_trajbatch(self):

        cpind = []
        alphaval = []
        indcp = np.load(self.rootpath+'trajp_value_indices.npy')
        for ind in range(indcp.shape[0]):
            cpind.append((indcp[ind][1],indcp[ind][2]))
            alphaval.append(indcp[ind][0])

        subset_trindices = []
        btlist = set()
        cps = os.listdir(self.rootpath+'trajp/')
        res = [ 0 for i in range(len(self.trainset)) ]
        cpv = [ [] for i in range(len(self.trainset)) ]
        repeat = [ 0 for i in range(len(self.trainset)) ]
        fea = {}
        testgr = {}
        dict_contrib = {}
        ind = 0
        

        print(len(cpv))
        #Computes scores for instances in B (Eq. 8)
        for ckpt in cps:
                # considering your ckpt name is 'epoch_<epoch value>_batch_<batch value>.pth'.
                net = torch.load(self.rootpath+'trajp/'+ckpt)
                subset_pickindices =[]
                ep = int(ckpt.split('_')[1].split('_')[0])
                bt = int(ckpt.split('_')[3].split('.')[0])
                alpha = alphaval[cpind.index((ep,bt))]
                start = ((self.bsize-1)*bt)+bt
                end = start + (self.bsize-1)

                for s in range(start,end+1):
                    if s>=len(self.trainset):
                        break
                    subset_trindices.append(s)
                    subset_pickindices.append(s)
                    cpv[s].append((ep,bt))


                subsetcp = torch.utils.data.Subset(self.trainset, subset_pickindices)
                trainsubloader = torch.utils.data.DataLoader(subsetcp, batch_size=1, shuffle=False, num_workers=2)
                trainbatchloader = torch.utils.data.DataLoader(subsetcp, batch_size=self.bsize, shuffle=False, num_workers=2)

                for testi, data in enumerate(trainbatchloader,0):
                    inputs, targets = data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    inputs = inputs.squeeze(1)
                    train_gradb = self.helperobj.get_grad(inputs,targets, net).unsqueeze(0)

                print("Alpha")
                print(alpha)
                for batch_idx, (inputs, targets) in enumerate(trainsubloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    inputs = inputs.squeeze(1)
                    ind = ind + 1
                    train_grad_single = self.helperobj.get_grad(inputs, targets, net).unsqueeze(0)
                    #tempf = test_gradb*train_grad_single
                    tempf = train_gradb*train_grad_single
                    #print(tempf)
                    if torch.sum(tempf)==0.0:
                        print("Batch")
                        print(train_gradb)
                        print("Single")
                        print(train_grad_single)
                    temp = (tempf + 0.5*(tempf*tempf))/len(trainsubloader)  #Element wise multiplication
                    repeat[subset_pickindices[batch_idx]]+=1
                    res[subset_pickindices[batch_idx]] += (alpha*(torch.sum(temp).item()))/repeat[subset_pickindices[batch_idx]] #score

        with open('./influence_scores.npy', 'wb') as f:
            np.save(f, np.array(res))

        return res

    ############ If computed data values (influence score, contribution vectors) are not to be used, the existing ones are removed and calculated again using selected trajectories ##############
    ############ Returns score and contribution vectors ##########
    def scorevalue(self):

        ########### If one has resumed, scores have to be recomputed. In order to avoid any eventuality, making retain_scores false explicitly############# 
        if self.resume:
            self.keeprem = False #If training is resumed from some point, data values/scores have to be recomputed
 
        if not self.keeprem:
            if os.path.exists('./influence_scores.npy'):
                os.remove('./influence_scores.npy')
            if os.path.exists('./dict_contrib.pkl'):
                os.remove('./dict_contrib.pkl')

            scores = self.score_trajbatch()

        else:
            if os.path.exists('influence_scores.npy'):
                scores = np.load('./influence_scores.npy')
            else:
                scores = None #retain_scores True even if no scores are kept recomputed
                
        return scores
