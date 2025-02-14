import json

params = {}

#Flags for Trajectory Selection
params['csel'] = True #True for executing CheckSel during training
params['resume'] = False #True for resuming training where it was left off

#Flags for Data Valuation
params['retain_scores'] = False #False for new/replaced computation of data values, True for using existing ones

#Flags for finding subset if required
params['findsubset'] = True #Flag for obtaining subset
params['simsel'] = False #Flag for using SubSel to obtain subset

#Hyperparameters during Trajectory Selection
params['trainbatch'] = 100 #Training data batch size
params['testbatch'] = 100 #Test data batch size
params['epochs'] = 200 #Training epochs
params['num_trajpoint'] = 480 #Number of to-be selected batches
params['num_freqcp'] = 50 #Frequency of updates at which training checkpoints will be saved
params['num_freqep'] = 20 #Frequency of epochs at which CheckSel algorithm will be executed
params['num_freqbatch'] = 10 #Frequency of batches at which CheckSel algorithm will be executed

#Hyperparameters during Data Valuation using CheckSel
params['featuredim'] = 512 #Feature vector dimension from the model to be used for computing neighbours

#Hyperparameters during finding subset
params['num_datapoints'] = 3700 #Number of to-be selected datapoints

#Path
params['root_dir'] = './main/'

with open("config.json", "w") as outfile:
    json.dump(params, outfile)
