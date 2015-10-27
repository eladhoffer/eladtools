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
  self.mask = self.mask or input.new()
  self.mask:resizeAs(input):copy(input)
  local dim = 2
  local size = math.floor(input:size(dim)*self.ratio)

  torch.sort(self.values,input, dim,true)

  self.mask:add(-self.values:narrow(dim,size+1,1):expandAs(input))
  torch.gt(self.mask, self.mask, 0)
  --self.mask:scatter(dim,indices,1):typeAs(input)
  torch.cmul(self.output, input, self.mask)
  return self.output
end

function SpatialBottleNeck:updateGradInput(input, gradOutput)
  torch.cmul(self.gradInput, gradOutput,self.mask)
  return self.gradInput
end
