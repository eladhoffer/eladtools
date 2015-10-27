require 'nn'
require 'cunn'
local GlobalDominantPooling, parent = torch.class('nn.GlobalDominantPooling', 'nn.Module')

function GlobalDominantPooling:__init(ratio)
  self.ratio = ratio or 0.25
  self.output = torch.Tensor()
  self.mask = torch.Tensor()
  self.buffer = torch.Tensor()
  self.gradInput = torch.Tensor()
end

function GlobalDominantPooling:updateOutput(input)
  torch.max(self.mask, self.mask, input, 2)
  local _,dominantFeat =  input:view(input:size(1),input:size(2),-1):sum(3):max(2)
  dominantFeat = dominantFeat:view(input:size(1), 1, 1, 1):expand(input:size(1), 1, input:size(3), input:size(4))


  torch.eq(self.mask, self.mask, dominantFeat)
  --self.mask:scatter(dim,indices,1):typeAs(input)

  self.mask:cdiv(self.mask:view(input:size(1),-1):sum(2):view(-1,1,1,1):expandAs(self.mask))

  torch.cmul(self.buffer, input, self.mask:expandAs(input))
  torch.sum(self.output, self.buffer:view(input:size(1),input:size(2),-1), 3)
  self.output = self.output:select(3,1)
  return self.output
end

function GlobalDominantPooling:updateGradInput(input, gradOutput)
  torch.cmul(self.gradInput, gradOutput:view(input:size(1),input:size(2),1,1):expandAs(input),self.mask:expandAs(input))
  return self.gradInput:expandAs(input)
end
