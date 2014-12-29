local Optimizer = torch.class('Optimizer')

function Optimizer:__init(...)
    xlua.require('torch',true)
    xlua.require('nn',true)
    local args = dok.unpack(
    {...},
    'Optimizer','Initialize an optimizer',
    {arg='Model', type ='table', help='Optimized model',req=true},
    {arg='Loss', type ='function', help='Loss function',req=true},
    {arg='Parameters', type = 'table', help='Model parameters - weights and gradients',req=false},
    {arg='OptFunction', type = 'function', help = 'Optimization function' ,req = true},
    {arg='OptState', type = 'table', help='Optimization configuration', default = {}, req=false},
    {arg='HookFunction', type = 'function', help='Hook function of type fun(y,yt,err)', req = false}
    )
    self.Model = args.Model
    self.Loss = args.Loss
    self.Parameters = args.Parameters
    self.OptFunction = args.OptFunction
    self.OptState = args.OptState
    self.HookFunction = args.HookFunction

    if self.Parameters == nil then
        self.Parameters = {}
        self.Weights, self.Gradients = self.Model:getParameters()
    else	
        self.Weights, self.Gradients = self.Parameters[1], self.Parameters[2] 
    end
end

function Optimizer:optimize(x,yt)
    local f_eval = function()
        self.Gradients:zero()
        local y = self.Model:forward(x)
        local err = self.Loss:forward(y,yt)
        local dE_dy = self.Loss:backward(y,yt)
        local value = nil
        self.Model:backward(x, dE_dy)
        if self.HookFunction then
            value = self.HookFunction(y,yt,err)
        end
        return err, self.Gradients
    end

    return value, self.OptFunction(f_eval, self.Weights, self.OptState)
end





