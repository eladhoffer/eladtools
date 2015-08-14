require 'torch'
require 'nn'
require 'cunn'
eladtools = {}
torch.include('eladtools', 'RecurrentLayer.lua')
torch.include('eladtools', 'EarlyStop.lua')
torch.include('eladtools', 'utils.lua')
torch.include('eladtools', 'Optimizer.lua')
torch.include('eladtools', 'ODCT.lua')
torch.include('eladtools', 'SpatialConvolutionDCT.lua')
torch.include('eladtools', 'SelectPoint.lua')
torch.include('eladtools', 'SpatialBottleNeck.lua')
torch.include('eladtools', 'SpatialNMS.lua')
