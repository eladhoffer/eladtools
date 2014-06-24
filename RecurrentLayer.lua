local RecurrentLayer, parent = torch.class('nn.RecurrentLayer', 'nn.Module')


function RecurrentLayer:__init(layers, iterations)
    self.modules = {}
    self.iterations = iterations
    self.layers = #layers
    self.saved_outputs = {}

    for j=1, self.layers do
        table.insert(self.modules, layers[j])

    end

    for i=1, self.iterations-1 do
        for j=1, self.layers do
            table.insert(self.modules, self.modules[j])

        end
    end
    self.gradInput = self.modules[1].gradInput
    self.output = self.modules[#self.modules].output
end

function RecurrentLayer:setIterations(iterations)
    local new_table = {}
    for j=1, self.layers do
        table.insert(new_table, self.modules[j])

    end
    self.modules = new_table
    self.iterations = iterations
    self.saved_outputs = {}
    for i=1, self.iterations-1 do
        for j=1, self.layers do
            table.insert(self.modules, self.modules[j])

        end
    end
    self.gradInput = self.modules[1].gradInput
    self.output = self.modules[#self.modules].output
end

function RecurrentLayer:size()
   return #self.modules
end

function RecurrentLayer:get(index)
   return self.modules[index]
end

function RecurrentLayer:updateOutput(input)
   local currentOutput = input
   for i=1,#self.modules do 
       currentOutput = self.modules[i]:updateOutput(currentOutput)
       if (#self.saved_outputs >= i) then
           self.saved_outputs[i]:copy(currentOutput)
       else
           self.saved_outputs[i] = currentOutput:clone()
       end
   end 
   self.output = currentOutput
   return currentOutput
end

function RecurrentLayer:updateGradInput(input, gradOutput)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      --currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
      currentGradOutput = currentModule:updateGradInput(self.saved_outputs[i], currentGradOutput)
      currentModule = previousModule
   end
   currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function RecurrentLayer:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accGradParameters(self.saved_outputs[i], currentGradOutput, scale)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end
   
   currentModule:accGradParameters(input, currentGradOutput, scale)
end

function RecurrentLayer:accUpdateGradParameters(input, gradOutput, lr)
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

function RecurrentLayer:zeroGradParameters()
  for i=1,#self.modules do
     self.modules[i]:zeroGradParameters()
  end
end

function RecurrentLayer:updateParameters(learningRate)
   for i=1,#self.modules do
      self.modules[i]:updateParameters(learningRate)
   end
end

function RecurrentLayer:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function RecurrentLayer:reset(stdv)
   for i=1,#self.modules do
      self.modules[i]:reset(stdv)
   end
end

function RecurrentLayer:parameters()
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

function RecurrentLayer:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.RecurrentLayer - ' .. tostring(self.iterations) .. ' iterations'
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
