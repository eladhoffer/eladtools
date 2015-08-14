
local PairwiseCriterion, parent = torch.class('nn.PairwiseCriterion', 'nn.Module')

function PairwiseCriterion:__init(criterion)
   parent.__init(self)
   self.criterion = criterion
   -- state
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.output = torch.Tensor()
end 
  
function PairwiseCriterion:updateOutput(input)
    self.output = self.criterion:forward(input[1],input[2])
    return self.output
end

function PairwiseCriterion:updateGradInput(input, gradOutput)
   self.gradInput[1]:resize(input[1]:size()):typeAs(input[1])
   self.gradInput[2]:resize(input[2]:size()):typeAs(input[2])
   self.gradInput[1] = self.criterion:updateGradInput(input[1],gradOutput)
     self.gradInput[2]:zero():add(-1, self.gradInput[1])
   return self.gradInput
end

-- save away Module:type(type) for later use.
PairwiseCriterion._parent_type = parent.type

function PairwiseCriterion:type(type)
   self:_parent_type(type)  -- Call the parent (Module) type function
   -- Now convert the left over table elements
   self.criterion:type(type)
   self.gradInput[1] = self.gradInput[1]:type(type)
   self.gradInput[2] = self.gradInput[2]:type(type)
   return self
end

