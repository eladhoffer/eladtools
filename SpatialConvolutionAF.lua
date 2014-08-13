local SpatialConvolutionAF, parent = torch.class('nn.SpatialConvolutionAF', 'nn.Module')

function SpatialConvolutionAF:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padding, partialSum)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1
   partialSum = partialSum or 0

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
   self.padding = padding or 0
   self.partialSum = partialSum

   self.weight = torch.Tensor(nInputPlane, kH, kW, nOutputPlane)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeightPartial = torch.Tensor(nInputPlane, kH, kW, nOutputPlane)
   self.gradWeight = torch.Tensor(nInputPlane, kH, kW, nOutputPlane)
   self.gradBias = torch.Tensor(nOutputPlane)
  
   self:reset()
end

function SpatialConvolutionAF:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function SpatialConvolutionAF:updateOutput(input)
   input.nn.SpatialConvolutionAF_updateOutput(self, input)
   local biasrep = self.bias:new():resize(self.bias:size(1),1,1,1):expandAs(self.output)
   self.biasrepc = self.biasrepc or biasrep.new()
   self.biasrepc:resizeAs(self.output):copy(biasrep)
   self.output:add(self.biasrepc)
   return self.output
end

function SpatialConvolutionAF:updateGradInput(input, gradOutput)
   input.nn.SpatialConvolutionAF_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialConvolutionAF:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   input.nn.SpatialConvolutionAF_accGradParameters(self, input, gradOutput, scale)
   if self.partialSum > 0 then
      self.gradWeight:add(self.gradWeightPartial:sum(1):resizeAs(self.gradWeight))
   end
   local sums = gradOutput:new():resize(gradOutput:size(1), gradOutput:size(2)*gradOutput:size(3)*gradOutput:size(4)):sum(2)
   self.gradBias:add(scale, sums)
end

-- this routine copies weight+bias from a regular SpatialConvolution module
function SpatialConvolutionAF:copy(sc)
   local weight = sc.weight:clone()
   weight:resize(sc.nOutputPlane, sc.nInputPlane * sc.kH * sc.kW)
   weight = weight:t():contiguous()
   weight:resize(sc.nInputPlane, sc.kH, sc.kW, sc.nOutputPlane)
   self.weight:copy(weight)
   self.bias:copy(sc.bias)
end

