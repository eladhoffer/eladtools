
require 'paths'

local EarlyStop = torch.class('EarlyStop')

function EarlyStop:__init(allowedBadIters, maxIterations)
    self.allowedBadIters = allowedBadIters or 5
    self.maxIterations = maxIterations or -1
    self.iteration = 0
    self.bestIteration = 0
    self.badStreak = 0
    self.stopped = false
end

function EarlyStop:update(currentError)
    self.iteration = self.iteration + 1

    if (self.iteration == 1 or currentError < self.lowestError) then
        self.badStreak = 0
        self.lowestError = currentError
        self.bestIteration = self.iteration
    else
        self.badStreak = self.badStreak + 1
    end

    self.lastError = currentError

    if (self.badStreak >= self.allowedBadIters) then
        self.stopped = true --stop
    else
        self.stopped = false --continue
    end

    if self.iteration == self.maxIterations then
        self.stopped = true
    end


    return self.stopped
end

function EarlyStop:reset()
	self.badStreak = 0
  self.iteration = 0
  self.stopped = false
end

function EarlyStop:stop()
    return self.stopped
end

function EarlyStop:lowest()
    return self.lowestError, self.bestIteration
end
