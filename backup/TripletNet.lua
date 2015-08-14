local TripletNet, parent = torch.class('nn.TripletNet', 'nn.Module')


function TripletNet:__init(net, num, dist)
    self.BatchMode = true
    self.nNets = num or 3
    self.Dist = dist or nn.PairwiseDistance(2)
    self.SubNet = nn.Sequential()
    if net then
        self.SubNet:add(net)
    end
    
    self.output = torch.Tensor()
    self:RebuildNet()
end

function TripletNet:RebuildNet()
    self.Net = nn.ParallelTable()
    self.Net:add(self.SubNet)
    for i=2, self.nNets do
        self.Net:add(self.SubNet:clone('weight','bias','gradWeight','gradBias'))
    end
end


function TripletNet:add(module)
    self.SubNet:add(module)
    self:RebuildNet()
end

function TripletNet:updateOutput(x)
    if self.BatchMode then
        self.BatchSize = x[1]:size(1)
    else
        self.BatchSize = 1
    end
    self.output:resize(self.BatchSize, self.nNets - 1):typeAs(x[1])
    self.sub_output = self.Net:updateOutput(x)
    for i=1, self.nNets-1 do
        self.output[{{},i}] = self.Dist:updateOutput({self.sub_output[1],self.sub_output[i+1]})
    end
    return self.output
end
function TripletNet:updateGradInput(x,gradOutput)
    self.DistGradInput = {}--torch.zeros(self.BatchSize):typeAs(x[1]),13}
    for i=1, self.nNets-1 do
        local dyi = gradOutput[{{},i}]
        if type(dyi) == 'number' then
            dyi = torch.Tensor({dyi}):typeAs(gradOutput)
        end
        local dEi = self.Dist:updateGradInput({self.sub_output[1],self.sub_output[i+1]},dyi)
        if self.DistGradInput[1] == nil then
            self.DistGradInput[1] = dEi[1]:clone()
        else
            self.DistGradInput[1]:add(dEi[1])
        end
        self.DistGradInput[i+1] = dEi[2]:clone()
    end
    return self.Net:updateGradInput(x, self.DistGradInput)
end

function TripletNet:accGradParameters(input, gradOutput, scale)
    self.Net:accGradParameters(input, self.DistGradInput, scale)
end


function TripletNet:training()
    self.Net:training()
end

function TripletNet:evaluate()
    self.Net:evaluate()
end

function TripletNet:getParameters(module)

    local w, g = self.SubNet:getParameters()
    self:RebuildNet()
    return w,g
end

function TripletNet:type(t)
    self.SubNet:type(t)
    self.Dist:type(t)
    self.output = self.output:type(t)
    self:RebuildNet()

end

