require 'nn'
require 'cunn'

function CPU2CUDA(model)
    local cuda_model = nn.Sequential()
    local Transposed = false
    for _,m in ipairs(model.modules) do
        if m.__typename == 'nn.SpatialConvolution' then
            if ( not Transposed) then 
                cuda_model:add(nn.Transpose({1,4},{1,3},{1,2}))
                Transposed = true
            end
            cuda_conv = nn.SpatialConvolutionCUDA(m.nInputPlane, m.nOutputPlane, m.kW, m.kH)
            cuda_conv.weight:copy(m.weight:transpose(1,4):transpose(1,2):transpose(2,3):cuda())
            cuda_conv.bias:copy(m.bias:cuda())
            cuda_model:add(cuda_conv)

        elseif m.__typename == 'nn.SpatialMaxPooling' then
            if (not Transposed) then 
                cuda_model:add(nn.Transpose({1,4},{1,3},{1,2}))
                Transposed = true
            end
            cuda_model:add(nn.SpatialMaxPoolingCUDA(m.kW, m.kH))

        elseif (m.__typename == 'nn.Reshape' or m.__typename == 'nn.SpatialZeroPadding')  then
            if (Transposed) then 
                cuda_model:add(nn.Transpose({1,4},{2,4},{3,4}))
                Transposed = false
            end
            cuda_model:add(m)

        else 
            cuda_model:add(m)
        end
    end
    cuda_model:cuda()
    return cuda_model
end


