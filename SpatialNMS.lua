require 'nn'
require 'cunn'
local SpatialNMS, parent = torch.class('nn.SpatialNMS', 'nn.Module')

function SpatialNMS:__init(ratio)
  self.ratio = ratio or 0.25
  self.output = torch.Tensor()
  self.mask = torch.Tensor()
  self.indices = torch.Tensor()
  self.gradInput = torch.Tensor()

end

function SpatialNMS:updateOutput(input)
  --if self.train then
    self.mask:typeAs(input):resizeAs(input):zero()
    self.indices:typeAs(input):resizeAs(input)
    local dim = 2

    _, self.indices = input:max(dim)

    self.mask:scatter(dim,self.indices,1)
    self.output = torch.cmul(input, self.mask)
  --else
  --  self.output = input
  --end
  return self.output
end

function SpatialNMS:updateGradInput(input, gradOutput)
  --if self.train then
    self.gradInput:typeAs(input):resizeAs(input)
    self.gradInput = torch.cmul(gradOutput,self.mask)
  --else
  --  self.gradInput = gradOutput
  --end
  return self.gradInput
end
