require 'nn'
require 'cutorch'

local GpuConcat, parent = torch.class('nn.GpuConcat', 'nn.Concat')

function GpuConcat:__init(dimension)
   parent.__init(self, dimension)
end

function GpuConcat:updateOutput(input)
   local outs = {}
   local n = #self.modules
   cutorch.reserveStreams(n)
   local prevStream = cutorch.getStream()
   local streamQueue = {}
   for i=1, n do
      cutorch.setStream(i)
      cutorch.streamWaitFor(i, {prevStream})
      table.insert(streamQueue, i)
      self.modules[i]:updateOutput(input)
   end
   cutorch.setStream(prevStream)
   cutorch.streamWaitFor(prevStream, streamQueue)

   for i=1,#self.modules do
      local currentOutput = self.modules[i].output
      outs[i] = currentOutput
      if i == 1 then
         self.size:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         self.size[self.dimension] = self.size[self.dimension] + currentOutput:size(self.dimension)
      end
   end
   self.output:resize(self.size)

   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = outs[i]
      self.output:narrow(self.dimension, offset, currentOutput:size(self.dimension)):copy(currentOutput)
      offset = offset + currentOutput:size(self.dimension)
   end
   return self.output
end

function GpuConcat:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   local n = #self.modules
   cutorch.reserveStreams(n)
   local prevStream = cutorch.getStream()
   local streamQueue = {}
   local offset = 1
   for i=1, n do
      cutorch.setStream(i)
      cutorch.streamWaitFor(i, {prevStream})
      table.insert(streamQueue, i)
      local currentOutput = self.modules[i].output
      self.modules[i]:updateGradInput(input, gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)))
      offset = offset + currentOutput:size(self.dimension)
   end
   cutorch.setStream(prevStream)
   cutorch.streamWaitFor(prevStream, streamQueue)

   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      local currentGradInput = module.gradInput

      if currentGradInput then -- if the module does not produce a gradInput (for example first layer), then ignore it and move on.
         if i==1 then
            self.gradInput:copy(currentGradInput)
         else
            self.gradInput:add(currentGradInput)
         end
      end
      offset = offset + currentOutput:size(self.dimension)
   end
   return self.gradInput
end

function GpuConcat:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local n = #self.modules
   cutorch.reserveStreams(n)
   local prevStream = cutorch.getStream()
   local streamQueue = {}
   local offset = 1
   for i=1, n do
      cutorch.setStream(i)
      cutorch.streamWaitFor(i, {prevStream})
      table.insert(streamQueue, i)
      local module = self.modules[i]
      local currentOutput = module.output
      module:accGradParameters(
         input,
         gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)),
         scale)
      offset = offset + currentOutput:size(self.dimension)
   end
   cutorch.setStream(prevStream)
   cutorch.streamWaitFor(prevStream, streamQueue)

end

function GpuConcat:accUpdateGradParameters(input, gradOutput, lr)
   assert('NYI')
end
