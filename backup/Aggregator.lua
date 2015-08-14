local Aggregator, parent = torch.class('nn.Aggregator', 'nn.Module')


function Aggregator:__init(inPlanes, outPlanes, kWH, Classes, type_conv)
    parent.__init(self)

    self.type_conv = type_conv or 'normal'
    self.inPlanes = inPlanes
    self.kWH = kWH
    self.Classes = Classes
    self.outPlanes = outPlanes
    self.outConv = torch.Tensor()
    if type_conv == 'normal' then
    self.ConvLayer = nn.SpatialConvolution(inPlanes, outPlanes, kWH, kWH)
end

        self.gradInput = self.ConvLayer.gradInput
end



function Aggregator:updateOutput(input)
    self.ConvLayer:updateOutput(input)
    self.outConv:resizeAs(self.ConvLayer.output):typeAs(self.ConvLayer.output):copy(self.ConvLayer.output)
    
    if not self.ClassLayer then
        if type_conv == 'normal' then
            self.sizeOut = outConv:size(3)
            self.ClassLayer = nn.Linear(self.sizeOut*self.sizeOut*self.outPlanes, Classes)
        end

    end

    local nframe = input:size(1)
    self.output:resize(nframe, self.outPlanes + self.Classes, self.sizeOut, self.sizeOut):typeAs(input)

    self.outConv:resize(nframe, self.sizeOut*self.sizeOut*self.outPlanes)
    self.ClassLayer:updateOutput(outConv)

    self.outConv:resizeAs(self.ConvLayer.output)
    self.output[{{},{1,self.outPlanes},{},{}}] = self.outConv
    for i=1, self.sizeOut do
        for j=1, self.sizeOut do
            self.output[{{},{self.outPlanes+1,self.outPlanes+self.Classes},{i},{j}}]:copy(self.ClassLayer.output)
        end
    end

    return self.output

end

function Aggregator:updateGradInput(input, gradOutput)
    self.outConv:resize(nframe, self.sizeOut*self.sizeOut*self.outPlanes)

    local classgradInput = self.ClassLayer:updateGradInput(self.outConv,gradOutput[{{},{self.outPlanes+1,self.outPlanes+self.Classes},{1},{1}}])
    self.outConv:resizeAs(self.ConvLayer.output)
    self.gradInput = self.ConvLayer:updateGradInput(input, classgradInput:resize(classgradInput:size(1),self.outPlanes, self.sizeOut, self.sizeOut) + gradOutput:narrow(2,1,self.outPlanes) )
    return self.gradInput
end

function Aggregator:accGradParameters(input, gradOutput, scale)
 self.outConv:resize(nframe, self.sizeOut*self.sizeOut*self.outPlanes)

    local classgradInput = self.ClassLayer:updateGradInput(self.outConv,gradOutput[{{},{self.outPlanes+1,self.outPlanes+self.Classes},{1},{1}}])
    self.outConv:resizeAs(self.ConvLayer.output)
    self.gradInput = self.ConvLayer:updateGradInput(input, classgradInput:resize(classgradInput:size(1),self.outPlanes, self.sizeOut, self.sizeOut) + gradOutput:narrow(2,1,self.outPlanes) )
    return self.gradInput


end

function Aggregator:accUpdateGradParameters(input, gradOutput, lr)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accUpdateGradParameters(sel.saved_outputs[i], currentGradOutput, lr)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end
   
   currentModule:accUpdateGradParameters(input, currentGradOutput, lr)
end

function Aggregator:zeroGradParameters()
  for i=1,#self.modules do
     self.modules[i]:zeroGradParameters()
  end
end

function Aggregator:updateParameters(learningRate)
   for i=1,#self.modules do
      self.modules[i]:updateParameters(learningRate)
   end
end

function Aggregator:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function Aggregator:reset(stdv)
   for i=1,#self.modules do
      self.modules[i]:reset(stdv)
   end
end

function Aggregator:parameters()
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   local w = {}
   local gw = {}
   --for i=1,#self.modules do
   for i=1,self.layers do
      local mw,mgw = self.modules[i]:parameters()
      if mw then
         tinsert(w,mw)
         tinsert(gw,mgw)
      end
   end
   return w,gw
end

function Aggregator:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.Aggregator - ' .. tostring(self.iterations) .. ' iterations'
   str = str .. ' {' .. line .. tab .. '[input'
   for j = 1, self.iterations do
   for i=1,self.layers do
      str = str .. next .. '(' .. i .. ')'
   end
   end
   str = str .. next .. 'output]'
   for i=1,self.layers do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end
