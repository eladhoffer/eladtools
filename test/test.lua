require 'nn'
require 'eladtools'
require 'cunn'
--
--CompareModel = nn.Sequential()
--CompareModel:add(nn.SpatialConvolution(5,5,3,3))
--CompareModel:add(nn.SpatialZeroPadding(1,1,1,1))
--CompareModel:cuda()
--
--RecModel  = nn.RecurrentLayer({nn.SpatialConvolution(5,5,3,3), nn.SpatialZeroPadding(1,1,1,1)},3 ):cuda()
--x = torch.rand(5,16,16):cuda()
--
--y1 = RecModel:forward(x)
--
--y2 = CompareModel:forward(x)

--params = RecModel:getParameters()
--
--
M = 16
C = 10
Agg = nn.Aggregator({nn.SpatialConvolution(M+C,M, 5,5)}, 
                    {nn.Reshape(M*6*6), nn.Linear(M*6*6,10), nn.LogSoftMax(), }
