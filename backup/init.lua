require 'torch'
require 'nn'
require 'cunn'
require 'libeladtools'
torch.include('eladtools', 'RecurrentLayer.lua')
torch.include('eladtools', 'EarlyStop.lua')
torch.include('eladtools', 'PairwiseCriterion.lua')
torch.include('eladtools', 'utils.lua')
torch.include('eladtools', 'DataProvider.lua')
torch.include('eladtools','SpatialClassifier4D.lua')
torch.include('eladtools','Optimizer.lua')
torch.include('eladtools','TripletNet.lua')
