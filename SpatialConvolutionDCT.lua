local SpatialConvolutionDCT, parent = torch.class('SpatialConvolutionDCT', 'nn.Module')

function SpatialConvolutionDCT:__init(conv_module)--nInputPlane, nOutputPlane, kW, kH, dW, dH, padding)
    self.conv = conv_module
   self:reset()
end

function SpatialConvolutionDCT:reset(stdv)
    if self.conv.nInputPlane>1 then
        self.conv.weight:copy(odct3dict(self.conv.nInputPlane,self.conv.kW,self.conv.kH,self.conv.nOutputPlane):narrow(2,1,self.conv.nOutputPlane):t())
    else
        self.conv.weight:copy(odct2dict(self.conv.kW,self.conv.kH,self.conv.nOutputPlane):narrow(2,1,self.conv.nOutputPlane):t())
    end
end

function SpatialConvolutionDCT:updateOutput(input)
    self.output = self.conv:updateOutput(input)
    return self.output
end

function SpatialConvolutionDCT:updateGradInput(input, gradOutput)
    self.gradInput = self.conv:updateGradInput(input,gradOutput)
    return self.gradInput
end

function SpatialConvolutionDCT:parameters()
return {self.conv.bias}, {self.conv.gradBias}
end

function SpatialConvolutionDCT:accGradParameters(input, gradOutput, scale)
end
function SpatialConvolutionDCT:type(t)
    self.conv:type(t)

end

--require 'cunn'
--
--local SpatialConvolutionDCT, parent = torch.class('SpatialConvolutionDCT', 'nn.SpatialConvolutionMM')
--
--function SpatialConvolutionDCT:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padding)
--   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padding)
--   
--   self:reset()
--end
--
--function SpatialConvolutionDCT:reset(stdv)
--   if stdv then
--      stdv = stdv * math.sqrt(3)
--   else
--      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
--   end
--   if nn.oldSeed then
--      self.weight:apply(function()
--         return torch.uniform(-stdv, stdv)
--      end)
--      self.bias:apply(function()
--         return torch.uniform(-stdv, stdv)
--      end)  
--   else
--      self.weight:uniform(-stdv, stdv)
--      self.bias:uniform(-stdv, stdv)
--   end
--   if self.nInputPlane>1 then
--       self.weight:copy(odct3dict(self.nInputPlane,self.kW,self.kH,self.nOutputPlane):narrow(2,1,self.nOutputPlane):t())
--   else
--
--       self.weight:copy(odct2dict(self.kW,self.kH,self.nOutputPlane):narrow(2,1,self.nOutputPlane):t())
--   end
--end
--
--function SpatialConvolutionDCT:parameters()
--return {self.bias}, {self.gradBias}
--end
--
--function SpatialConvolutionDCT:accGradParameters(input, gradOutput, scale)
--end
