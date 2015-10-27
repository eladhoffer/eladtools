
local SelectPoint, parent = torch.class('nn.SelectPoint', 'nn.Module')
function SelectPoint:__init(dimension, x,y)
  parent.__init(self)
  self.dimension = dimension
  self.y = y
  self.x= x
end
function SelectPoint:updateOutput(input)
  local output = input:select(self.dimension,self.y):select(self.dimension,self.x)
  self.output:resizeAs(output)
  return self.output:copy(output)
end
function SelectPoint:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)
  self.gradInput:zero()
  self.gradInput:select(self.dimension,self.y):select(self.dimension,self.x):copy(gradOutput)
  return self.gradInput
end
