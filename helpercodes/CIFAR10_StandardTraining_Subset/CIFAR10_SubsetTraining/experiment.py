from subsetpack.dataset import Dataset
from subsetpack.model import Model
from subsetpack.run import TrajSel
from subsetpack.helper import HelperFunc
from subsetpack.scoreval_datasel import DataValueCheckSel
from subsetpack.attack import Attack
import config_create
import os
import json

def main():

	############ Run the config file to create a dictionary ##########
	#os.system('python config_create.py')
	
	with open("config.json", "r") as fp:
		confdata = json.load(fp) #holds the various configurable parameters needed ahead

	########### Defining dataset class for loading the required data #############
	dataobj = Dataset(confdata)
	trainloader, testloader, trainset,testloader_s = dataobj.load_data()

	########### Defining model class for loading the model architecture ###########
	modelobj = Model()
	model = modelobj.ResNet18()

	########### Trains the model on the dataset and saves uniformly spaced trajectory points(model parameters)
	########### Or selected trajectories along with their importance weights using CheckSel
	helpobj = HelperFunc(trainloader,testloader,model,confdata)
	trajobj = TrajSel(trainset,trainloader,testloader,model,helpobj,confdata)
	trajobj.fit()

	########## Computes value for all training datapoints using selected trajectories from CheckSel #########

	#dvalueobj = DataValueCheckSel(trainset,testloader,model,helpobj,confdata)
	#scores = dvalueobj.scorevalue()
if __name__=='__main__':
	main()
