require 'cunn'

local SpatialConvolutionDCT, parent = torch.class('SpatialConvolutionDCT', 'nn.SpatialConvolutionMM')

function SpatialConvolutionDCT:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padding)
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padding)
   
   self:reset()
end

function SpatialConvolutionDCT:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)  
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
   if self.nInputPlane>1 then
       self.weight:copy(odct3dict(self.nInputPlane,self.kW,self.kH,self.nOutputPlane):narrow(2,1,self.nOutputPlane):t())
   else

       self.weight:copy(odct2dict(self.kW,self.kH,self.nOutputPlane):narrow(2,1,self.nOutputPlane):t())
   end
end

function SpatialConvolutionDCT:parameters()
return {self.bias}, {self.gradBias}
end

--function SpatialConvolutionDCT:accGradParameters(input, gradOutput, scale)
--end
