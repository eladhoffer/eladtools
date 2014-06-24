local SpatialSoftMax,parent = torch.class('nn.SpatialSoftMax', 'nn.Module')

function SpatialSoftMax:__init(constant)
   parent.__init(self)
   self.constant_present = false
   self.constant = constant -- a constant added to the exponential sum
   if self.constant then
      print('Using exp('..self.constant
..') as additional denomimator constant in SpatialSoftMax')
      self.constant_present = true
   end
end

function SpatialSoftMax:updateOutput(input)
   nn.SpatialSoftMax_updateOutput(self, input)
   self.output = input
   return self.output
end

function SpatialSoftMax:updateGradInput(input, gradOutput)
   nn.SpatialSoftMax_updateGradInput(self, input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end
