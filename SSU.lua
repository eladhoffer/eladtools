local SSU, parent = torch.class('nn.SSU', 'nn.Module')
function SSU:__init(scale)
  parent.__init(self)
  self.scale = scale or 1
  self.buffer = torch.Tensor()

end
function SSU:updateOutput(input)
  self.buffer = self.buffer or input.new()
  self.buffer:resizeAs(input):copy(input):mul(self.scale):exp():add(1)
  self.output:resizeAs(input):copy(self.buffer):log():div(self.scale)
  return self.output
end
function SSU:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):copy(self.buffer):pow(-1):mul(-1):add(1):cmul(gradOutput)
  return self.gradInput
end
