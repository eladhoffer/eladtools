local Optimizer = torch.class('Optimizer')

function Optimizer:__init(...)
    xlua.require('torch',true)
    xlua.require('nn',true)
    local args = dok.unpack(
    {...},
    'Optimizer','Initialize an optimizer',
    {arg='Model', type ='table', help='Optimized model',req=true},
    {arg='Loss', type ='function', help='Loss function',req=true},
    {arg='Regime', type ='table', help='Training Regime Table',default = nil},
    {arg='L1Coeff', type ='number', help='L1 Regularization coeff',default=0},
    {arg='GradClip', type ='number', help='Gradient clipping value. zero for off',default=0},
    {arg='GradRenorm', type ='number', help='Gradient renorm value. zero for off',default=0},
    {arg='GradRescale', type ='number', help='Gradient rescale value. zero for off',default=0},
    {arg='Parameters', type = 'table', help='Model parameters - weights and gradients',req=false},
    {arg='OptFunction', type = 'function', help = 'Optimization function' ,req = true},
    {arg='OptState', type = 'table', help='Optimization configuration', default = {}, req=false},
    {arg='HookFunction', type = 'function', help='Hook function of type fun(y,yt,err)', req = false}
    )
    for x,val in pairs(args) do
        self[x] = val
    end
    if self.Parameters == nil then
        self.Parameters = {}
        self.Weights, self.Gradients = self.Model:getParameters()
    else
        self.Weights, self.Gradients = self.Parameters[1], self.Parameters[2]
    end
end

function Optimizer:optimize(x,yt)
  local y, err, value
  local f_eval = function()
    self.Model:zeroGradParameters()
    y = self.Model:forward(x)
    err = self.Loss:forward(y,yt)
    local dE_dy = self.Loss:backward(y,yt)
    self.Model:backward(x, dE_dy)
    if self.HookFunction then
      value = self.HookFunction(y,yt,err)
    end

    if self.L1Coeff>0 then
      self.Gradients:add(torch.sign(self.Weights):mul(self.L1Coeff))
    end

    if self.GradClip > 0 then
      self.Gradients:clamp(-self.GradClip, self.GradClip)
    end

    if self.GradRenorm > 0 then
      local norm = self.Gradients:norm()
      if norm > self.GradRenorm then
        local shrink = self.GradRenorm / norm
        self.Gradients:mul(shrink)
      end
    end

    if self.GradRescale > 0 then
      local norm = math.max(self.Gradients:max(), -self.Gradients:min())
      if norm > self.GradRescale then
        local shrink = self.GradRescale / norm
        self.Gradients:mul(shrink)
      end
    end

    return err, self.Gradients
  end
  local opt_value = self.OptFunction(f_eval, self.Weights, self.OptState)
  return y, err,value, opt_value
end

function Optimizer:updateRegime(epoch, verbose)
  if self.Regime then
    if self.Regime.epoch then
      for epochNum,epochVal in pairs(self.Regime['epoch']) do
        if epochVal == epoch then
          for optValue,_ in pairs(self.Regime) do
            if self.OptState[optValue] then
              if verbose then
                print(optValue,': ',self.OptState[optValue], ' -> ', self.Regime[optValue][epochNum])
              end
              self.OptState[optValue] = self.Regime[optValue][epochNum]
            end
          end
        end
      end
    end
  end
end
--function Optimizer:optimStates(opts)--opts must be of for {{weight = optimState, bias = optimState} .... }
--    for i, optimState in ipairs(opts) do
--local weight_size = self.Weights:size(1)
--local learningRates = torch.Tensor(weight_size):fill(self.OptState.learningRate)
--local weightDecays = torch.Tensor(weight_size):fill(self.OptState.weightDecay)
--local counter = 0
--for i, layer in ipairs(model.modules) do
--      local weight_size = layer.weight:size(1)*layer.weight:size(2)
--      learningRates[{{counter+1, counter+weight_size}}]:fill(1)
--      weightDecays[{{counter+1, counter+weight_size}}]:fill(wds)
--      counter = counter+weight_size
--      local bias_size = layer.bias:size(1)
--      learningRates[{{counter+1, counter+bias_size}}]:fill(2)
--      weightDecays[{{counter+1, counter+bias_size}}]:fill(0)
--      counter = counter+bias_size
--   end
--end
