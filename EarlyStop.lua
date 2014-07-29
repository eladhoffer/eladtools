
require 'paths'

local EarlyStop = torch.class('EarlyStop')

function EarlyStop:__init(AllowedBadIters, MaximumIterations, Model, path)
    self.AllowedBadIters = AllowedBadIters or 5
    self.MaximumIterations = MaximumIterations or -1
    self.Iter = 0
    self.LastError = 100
    self.LowestError = 100
    self.BestIter = 0
    self.BadStreak = 0
    self.Stopped = false
    self.Model = Model
    self.path = path
end

function EarlyStop:SaveNet()
    local filename
    filename = paths.concat(self.path, 'BestNetEarlyStop' )
    torch.save(filename,self.Model)
end

function EarlyStop:Update(Success)
    self.Iter = self.Iter + 1
        
    local CurrentError = 100 - Success

    if (CurrentError < self.LowestError) then
        self.BadStreak = 0
        self.LowestError = CurrentError
        self.BestIter = self.Iter
        if self.Model then
            self:SaveNet()
        end
    else
        self.BadStreak = self.BadStreak + 1
    end

    self.LastError = CurrentError

    if (self.BadStreak >= self.AllowedBadIters) then
        self.Stopped = true --stop
    else
        self.Stopped = false --continue
    end

    if self.Iter == self.MaximumIterations then
        self.Stopped = true
    end


    return self.Stopped
end

function EarlyStop:Stop()
    return self.Stopped
end

function EarlyStop:PrintStatus()
    if self.Stopped then
        print('Stopped after ' .. tostring(self.Iter) .. ' iterations, with Error: ' .. tostring(self.LastError))
    else
        print(tostring(self.Iter) .. ' iterations, Current Error: ' .. tostring(self.LastError))
    end
    print('Best Iteration: ' .. tostring(self.BestIter) .. ' with ' .. tostring(self.LowestError))
end

