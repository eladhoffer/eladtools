require 'nn'
require 'cunn'
local SpatialBottleNeck, parent = torch.class('nn.SpatialBottleNeck', 'nn.Module')

function SpatialBottleNeck:__init(ratio)
  self.ratio = ratio or 0.25
  self.output = torch.Tensor()
  self.mask = torch.Tensor()
  self.values = torch.Tensor()
  self.gradInput = torch.Tensor()
end

function SpatialBottleNeck:updateOutput(input)
  self.mask:typeAs(input):resizeAs(input):copy(input)
  self.values:typeAs(input):resizeAs(input):zero()
  local dim = 2
  local size = math.floor(input:size(dim)*self.ratio)


  --indices = indices:narrow(dim,1,self.ratio)
  self.values = input:sort(dim,true)

  self.mask:add(-self.values:narrow(dim,size+1,1):expandAs(input))
  self.mask:copy(self.mask:gt(0))
  --self.mask:scatter(dim,indices,1):typeAs(input)
  self.output = torch.cmul(input, self.mask)
  return self.output
end

function SpatialBottleNeck:updateGradInput(input, gradOutput)
  self.gradInput:typeAs(input):resizeAs(input)
  self.gradInput = torch.cmul(gradOutput,self.mask)
  return self.gradInput
end
