local Classifier, parent = torch.class('nn.SpatialClassifier4D', 'nn.Module')

function Classifier:__init(classifier)
   parent.__init(self)
   -- public:
   self.classifier = classifier or nn.Sequential()
   self.spatialOutput = true
   -- private:
   self.inputF = torch.Tensor()
   self.inputT = torch.Tensor()
   self.outputF = torch.Tensor()
   self.output = torch.Tensor()
   self.gradOutputF = torch.Tensor()
   self.gradOutputT = torch.Tensor()
   self.gradInputF = torch.Tensor()
   self.gradInput = torch.Tensor()
   -- compat:
   self.modules = {self.classifier}
end

function Classifier:add(module)
   self.classifier:add(module)
end

function Classifier:updateOutput(input)
   -- get dims:
   if input:nDimension() ~= 4 then
      error('<nn.SpatialClassifier> input should be 4D: NxKxHxW')
   end
   local N = input:size(1)
   local K = input:size(2)
   local H = input:size(3)
   local W = input:size(4)
   local HW = H*W
   local NHW = N*H*W

   -- transpose input:
   self.inputF:set(input):resize(N,K,HW)
   self.inputT:resize(N,HW,K):copy(self.inputF:transpose(2,3))
   self.inputT:resize(NHW,K)

   -- classify all locations:
   self.outputT = self.classifier:updateOutput(self.inputT)

   if self.spatialOutput then
      -- transpose output:
      local O = self.outputT:size(2)
      self.outputT:resize(N,HW,O)
      self.outputF:resize(N,O,HW):copy(self.outputT:transpose(2,3))
      self.output:set(self.outputF):resize(N,O,H,W)
   else
      -- leave output flat:
      self.output = self.outputT
   end
   return self.output
end

function Classifier:updateGradInput(input, gradOutput)
   -- get dims:
   local N = input:size(1)
   local K = input:size(2)
   local H = input:size(3)
   local W = input:size(4)
   local HW = H*W
   local NHW = N*H*W
   local O = gradOutput:size(1)

   -- transpose input
   self.inputF:set(input):resize(N,K,HW)
   self.inputT:resize(N,HW,K):copy(self.inputF:transpose(2,3))
   self.inputT:resize(NHW,K)

   if self.spatialOutput then
      -- transpose gradOutput
      self.gradOutputF:set(gradOutput):resize(N,O,HW)
      self.gradOutputT:resize(N,HW,O):copy(self.gradOutputF:transpose(2,3))
      self.gradOutputT:resize(NHW,O)
   else
      self.gradOutputT = gradOutput
   end

   -- backward through classifier:
   self.gradInputT = self.classifier:updateGradInput(self.inputT, self.gradOutputT)

   -- transpose gradInput
   self.gradInputT:resize(N,HW,K)
   self.gradInputF:resize(N,K,HW):copy(self.gradInputT:transpose(2,3))
   self.gradInput:set(self.gradInputF):resize(N,K,H,W)
   return self.gradInput
end

function Classifier:accGradParameters(input, gradOutput, scale)
   -- get dims:
   local N = input:size(1)
   local K = input:size(2)
   local H = input:size(3)
   local W = input:size(4)
   local HW = H*W
   local NHW = N*H*W
   local O = gradOutput:size(1)

   -- transpose input
   self.inputF:set(input):resize(N,K,HW)
   self.inputT:resize(NHW,K):copy(self.inputF:transpose(2,3):resize(NHW,K))

   if self.spatialOutput then
      -- transpose gradOutput
      self.gradOutputF:set(gradOutput):resize(N,O,HW)
      self.gradOutputT:resize(NHW,O):copy(self.gradOutputF:transpose(2,3):resize(NHW,O))
   else
      self.gradOutputT = gradOutput
   end

   -- backward through classifier:
   self.classifier:accGradParameters(self.inputT, self.gradOutputT, scale)
end

function Classifier:zeroGradParameters()
   self.classifier:zeroGradParameters()
end

function Classifier:updateParameters(learningRate)
   self.classifier:updateParameters(learningRate)
end
