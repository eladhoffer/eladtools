require 'eladtools'

model = nn.Sequential()
model:add(nn.SpatialConvolution(3,16,5,5))
model:add(nn.SpatialBatchNormalization(16,nil,nil,false))
model:add(nn.SpatialConvolution(16,8,5,5))
--model:add(nn.View(8*2*2):setNumInputDims(3))
--model:add(nn.BatchNormalization(8*2*2,nil,nil,false))
--model:add(nn.Linear(8*2*2,10))

model:get(2).running_mean:normal():mul(10)
model:get(2).running_std:normal():mul(10)
model:get(1).weight:mul(100)
x= torch.rand(1,3,10,10):mul(100)


model:evaluate()
y = model:forward(x)


reduced = RemoveBN(model)

y2 = reduced:forward(x)

print(model)
print(reduced)
print(torch.dist(y,y2))
