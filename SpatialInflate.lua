local SpatialInflate, parent = torch.class('nn.SpatialInflate', 'nn.Module')

function SpatialInflate:__init(height, width)
    parent.__init(self)
    self.height = height
    self.width = width
end

function SpatialInflate:updateOutput(input)
    -- sizes
    local h = self.height
    local w = self.width

    if input:dim() == 3 then
        if w < 1 or h < 1 then error('input is too small') end
        self.output:resize(input:size(1), h, w)
        self.output:zero()
       -- copy input to output
        c_output:copy(c_input)
    elseif input:dim() == 4 then
        -- sizes
        local h = self.height
        local w = self.width
        if w < 1 or h < 1 then error('input is too small') end
        self.output:resize(input:size(1), input:size(2), h, w)
        self.output:zero()
    else
        error('input must be 3 or 4-dimensional')
    end
    return self.output
end

function SpatialInflate:updateGradInput(input, gradOutput)
   if input:dim() == 3 then 
      self.gradInput:resizeAs(input):zero()
      -- crop gradInput if necessary
      local cg_input = self.gradInput
      if self.pad_t < 0 then cg_input = cg_input:narrow(2, 1 - self.pad_t, cg_input:size(2) + self.pad_t) end
      if self.pad_b < 0 then cg_input = cg_input:narrow(2, 1, cg_input:size(2) + self.pad_b) end
      if self.pad_l < 0 then cg_input = cg_input:narrow(3, 1 - self.pad_l, cg_input:size(3) + self.pad_l) end
      if self.pad_r < 0 then cg_input = cg_input:narrow(3, 1, cg_input:size(3) + self.pad_r) end
      -- crop gradOutout if necessary
      local cg_output = gradOutput
      if self.pad_t > 0 then cg_output = cg_output:narrow(2, 1 + self.pad_t, cg_output:size(2) - self.pad_t) end
      if self.pad_b > 0 then cg_output = cg_output:narrow(2, 1, cg_output:size(2) - self.pad_b) end
      if self.pad_l > 0 then cg_output = cg_output:narrow(3, 1 + self.pad_l, cg_output:size(3) - self.pad_l) end
      if self.pad_r > 0 then cg_output = cg_output:narrow(3, 1, cg_output:size(3) - self.pad_r) end
      -- copy gradOuput to gradInput
      cg_input:copy(cg_output)
   elseif input:dim() == 4 then
      self.gradInput:resizeAs(input):zero()
      -- crop gradInput if necessary
      local cg_input = self.gradInput
      if self.pad_t < 0 then cg_input = cg_input:narrow(3, 1 - self.pad_t, cg_input:size(3) + self.pad_t) end
      if self.pad_b < 0 then cg_input = cg_input:narrow(3, 1, cg_input:size(3) + self.pad_b) end
      if self.pad_l < 0 then cg_input = cg_input:narrow(4, 1 - self.pad_l, cg_input:size(4) + self.pad_l) end
      if self.pad_r < 0 then cg_input = cg_input:narrow(4, 1, cg_input:size(4) + self.pad_r) end
      -- crop gradOutout if necessary
      local cg_output = gradOutput
      if self.pad_t > 0 then cg_output = cg_output:narrow(3, 1 + self.pad_t, cg_output:size(3) - self.pad_t) end
      if self.pad_b > 0 then cg_output = cg_output:narrow(3, 1, cg_output:size(3) - self.pad_b) end
      if self.pad_l > 0 then cg_output = cg_output:narrow(4, 1 + self.pad_l, cg_output:size(4) - self.pad_l) end
      if self.pad_r > 0 then cg_output = cg_output:narrow(4, 1, cg_output:size(4) - self.pad_r) end
      -- copy gradOuput to gradInput
      cg_input:copy(cg_output)
   else
      error('input must be 3 or 4-dimensional')
   end
   return self.gradInput
end
