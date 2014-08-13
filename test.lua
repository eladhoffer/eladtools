require 'eladtools'


m = nn.SpatialConvolutionAF(3,32,7,7):cuda()

x= torch.rand(3,32,32,1):cuda()

y = m:forward(x)
