require 'DataProvider'
require 'utils'
require 'cutorch'
--data = DataProvider('/home/ehoffer/Datasets/Cache', {3,256,256})
--if not data:ItemsLoaded() then
--data:GenerateFilenames('/home/ehoffer/Datasets/ImageNet/Attributes/',true)
--end
--        -- swap elements
local LabelFaceBackground = function(Item)
    if string.find(Item.Filename, "Face") then
        return 1
    else
        return 0
    end
end
local PreProcess = function(Img)
    return PadTensor(CropCenter(Img,100),100,100)
end

data = DataProvider('../../Face/Cache', {3,100,100})
if not data:ItemsLoaded() then
    data:GenerateFilenames({'/home/ehoffer/ehoffer/Datasets/Images/Faces/PubFig/', 
                            '/home/ehoffer/ehoffer/Datasets/Images/Backgrounds/'})
    data:LabelItems(LabelFaceBackground)
    data:ShuffleItems()
    data:SaveItemsList()
end
B = data:InitBatch(473)
b = data:InitMiniBatch(128,'torch.CudaTensor')

data:AddPrepFunc(PreProcess)

data:GetNextBatch()

local n=1
while data:GetNextMiniBatch(true) do
    --print('Batch number', n)
    print('Current Item', data:CurrentItemCount())
    n = n +1
end

