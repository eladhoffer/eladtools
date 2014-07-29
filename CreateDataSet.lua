require 'paths'
require 'image'

package.path = package.path .. ';/home/ehoffer/tools/?.lua;'
require 'image_manipulation'
PATH = '../iccv09Data/'

SizeX = 320
SizeY = 240
NumFiles = 716
Labels = torch.IntTensor(NumFiles, SizeY, SizeX)
Images = torch.DoubleTensor(NumFiles, 3, SizeY, SizeX)
NumImage = torch.IntTensor(NumFiles)
local num = 1
for f in paths.files(PATH .. 'images/') do
    local filename = PATH .. 'images/' .. f
    if paths.filep(filename) then
        local f_bn = paths.basename(filename,'.jpg')
        print(f_bn)
        local img = image.loadJPG(filename)
        local size = img:size()
        if (size[2] ~= SizeX) or (size[3] ~= SizeY) then
            img = PadTensor(img, SizeY, SizeX, 0)
        end
        Images[num] = img 
        NumImage[num] = tonumber(f_bn)
        local labels_filename = PATH .. 'labels/' .. f_bn .. '.regions.txt'

        local file = torch.DiskFile(labels_filename, 'r')
        local label = torch.IntTensor(size[2], size[3])
        for y=1,size[2] do
            for x=1,size[3] do
                label[y][x] = file:readInt()
            end
        end
        file:close()
        Labels[num] = PadTensor(label, SizeY, SizeX, -1)
        num = num + 1
    end
end

DataSet = {Labels = Labels, Images = Images, NumImage = NumImage}
torch.save('DataSet', DataSet)
