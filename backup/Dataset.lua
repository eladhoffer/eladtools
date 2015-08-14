--require 'image'
require 'torch'
package.path = package.path .. ';/home/ehoffer/tools/?.lua;'
require 'image_manipulation'

function ClassVec(num, Max)
    local Max = Max or 8
    local vec = torch.DoubleTensor(Max):zero()
    if (num<0) or (num+1>Max) then
        return vec
    end
    vec[num+1] = 1
    return vec
end

function Normalize(Data)
    local N = Data:size(1)
    for i=1,N do
        local Mean = Data[i]:mean()
        local Std = Data[i]:std()
        Data[i] = (Data[i] - Mean)/Std
    end
    return Data
end

function Quantize(Data, QLevel)
    Data:mul(QLevel):floor():mul(1/QLevel)
end

function SamplePatch(img, labels, x, y, PatchSize)
    local Side = (PatchSize-1)/2
    --local patch = image.crop(img,x-Side,y-Side,x+Side+1,y+Side+1)
    local patch
    local startx = math.max(1,x-Side)
    local starty = math.max(1,y-Side)
    local endx
    local endy
    local Mean =opt.Mean or 0
    local Std = opt.Std or 1
    if img:dim()==2 then
        endx = math.min(img:size(2), x+Side)
        endy = math.min(img:size(1), y+Side)
        patch = torch.Tensor(PatchSize, PatchSize):fill(-Mean/Std)--:zero()
        patch[{{1,endy-starty+1},{1,endx-startx+1}}] = img[{{starty,endy},{startx,endx}}]
    else
        endx = math.min(img:size(3), x+Side)
        endy = math.min(img:size(2), y+Side)
        patch = torch.Tensor(img:size(1),PatchSize, PatchSize):zero()
        patch[{{},{1,endy-starty+1},{1,endx-startx+1}}] = img[{{},{starty,endy},{startx,endx}}]
    end
    return patch, labels[y][x]+1

end

function GeneratePatches(num_patches, limit_num, x_limit, y_limit)
    local limit_num = limit_num or 572
    local x_limit = x_limit or 60
    local y_limit = y_limit or 60
    local imageSize = {240, 320}
    local FullData = torch.load('DataSet')
    local Patches = torch.DoubleTensor(num_patches, 3,121,121)
    local Labels = torch.DoubleTensor(num_patches)
    for num=1,num_patches do
        local pic = math.random(limit_num)
        local x = math.random(imageSize[2]-2*x_limit) + x_limit -1
        local y = math.random(imageSize[1]-2*y_limit) + y_limit -1
        local patch
        local class
        patch, class = SamplePatch(FullData.Images[pic], FullData.Labels[pic], x,y, 121)
        Patches[num]:copy(patch)
        Labels[num] = class
    end
    local data = {Patches = Patches, Labels = Labels}
    torch.save('Patches',data)

end

function ExtractPatchesList(data, list)
    local num_patches = list:size(1)
    local Patches = torch.DoubleTensor(num_patches, 3,121,121)
    local Labels = torch.DoubleTensor(num_patches)
    for num=1,num_patches do
        local pic = data.Images[list[num][1]]
        local x = list[num][2]
        local y = list[num][3]
        local patch
        local class
        patch, class = SamplePatch(pic, data.Labels[list[num][1]], x,y, 121)
        Patches[num]:copy(patch)
        Labels[num] = class
    end
    local data = {Patches = Patches, Labels = Labels}
    return data
end
function ExtractPatchEntry(data, entry)
    return SamplePatch(data.Images[entry[1]], data.Labels[entry[1]], entry[2], entry[3], 121)
end

function DownSample(img)
    local Oimg = torch.Tensor(img:size(1),img:size(2),img:size(3)):copy(img)
    local sz = torch.LongStorage({Oimg:size(1),25,25})
    local DownSampled = torch.Tensor()
    local stride = torch.LongStorage({Oimg:stride(1),Oimg:stride(2)*5,Oimg:stride(3)*5})
    local stg = Oimg:storage()
    DownSampled:set(stg,1,sz,stride)
    return DownSampled
end

function ExtractLabelsEntry(data, entry)
    local Labels = data.Labels[entry[1]]:reshape(1,240,320):float()
    local Labels_region = SamplePatch(Labels, data.Labels[entry[1]], entry[2], entry[3], 121)+1
    local DownLabels = DownSample(Labels_region:resize(1,121,121))--torch.IntTensor(25,25)
    DownLabels[DownLabels:lt(1)] = 1

    return DownLabels
end

function ExtractDiffPatchEntry(data, entry, dx, dy, rand)
    local dx = dx or 1
    local dy = dx or 1
    local n
    if rand then
        repeat
            n = math.random(4)
            if n==1 then
                dx = 1
                dy = 0
            elseif n==2 then
                dy = 1
                dx = 0 
            elseif n==3 then
                dy = -1
                dx = 0 
            elseif n==4 then
                dy = 0
                dx = -1
            end
        until (entry[2] + dx <= 260) and (entry[2] + dx > 60) and (entry[3] + dy <= 180) and (entry[3] + dy > 60)
    end 
    local patch1
    local label
    patch1, label = SamplePatch(data.Images[entry[1]], data.Labels[entry[1]], entry[2], entry[3], 121)
    local patch2 = SamplePatch(data.Images[entry[1]], data.Labels[entry[1]], entry[2] + dx, entry[3] + dy, 121)

    return patch1-patch2, label, n
end

function ExtractDirectionDiffPatchEntry(data, entry)
    local patch 
    local label
    local d
    patch, label, d = ExtractDiffPatchEntry(data, entry ,0,0 ,true)
    local direction = torch.IntTensor(4):zero()
    direction[d] = 1
    local new_label = torch.cat(label,direction)

    return patch, new_label
end

function CreateRandomList(data, start_img, end_img, length)
    --Chosen values - training 1-572,   test 573-715
    local list = torch.IntTensor(length, 4)
    for i=1,length do
        repeat
            list[i][1] = math.random(start_img, end_img)
            list[i][2] = math.random(1,320)
            list[i][3] = math.random(1,240)
            list[i][4] = data.Labels[list[i][1]][list[i][3]][list[i][2]]
        until list[i][4]  > -1
    end
    return list
end
function CreateRandomListLabelSubset(data, start_img, end_img, length,labels)
    --Chosen values - training 1-572,   test 573-715
    local list = torch.IntTensor(length, 4)
    local function fromLabels(label)
        for l=1,#labels do
            if label==labels[l] then
                return true
            end
        end
        return false
    end

    for i=1,length do
        repeat
            list[i][1] = math.random(start_img, end_img)
            list[i][2] = math.random(1,320)
            list[i][3] = math.random(1,240)
            list[i][4] = data.Labels[list[i][1]][list[i][3]][list[i][2]]
        until fromLabels(list[i][4])
    end
    return list
end

--
--function CreateRandomList(data, start_img, end_img, length)
--    --Chosen values - training 1-572,   test 573-715
--    local list = torch.IntTensor(length, 4)
--    for i=1,length do
--        repeat
--            list[i][1] = math.random(start_img, end_img)
--            list[i][2] = math.random(61,260)
--            list[i][3] = math.random(61,180)
--            list[i][4] = data.Labels[list[i][1]][list[i][3]][list[i][2]]
--        until list[i][4]  > -1
--    end
--    return list
--end

function CreateFullList(Data, start_img, end_img)
    local i = 1
    local list = torch.IntTensor((end_img-start_img + 1)*320*240, 4)
    for k=start_img,end_img do
        local CurrLabel = Data.Labels[k]
        local CurrImage = Data.Images[k]
        for m=1,CurrLabel:size(1) do
            for n=1,CurrLabel:size(2) do
                if CurrLabel[m][n]>=0 then
                    list[i][1] = k
                    list[i][2] = n
                    list[i][3] = m
                    list[i][4] = CurrLabel[m][n]
                    if CurrLabel[m][n]>=0 then
                        i = i+1
                    end
                end
            end
        end
    end
    return list[{{1,i-1},{}}]
end

function CountStats(Data)
    local N = 1--Data:size(1)
    local Count = torch.zeros(9)
    local AllCount = 0
    local list = torch.IntTensor(N*199*119)
    local curr = 1
    for i=1,N do
        local CurrLabel = Data[i]
        for m=61,CurrLabel:size(1)-61 do
            for n=61,CurrLabel:size(2)-61 do
                --if CurrLabel[m][n]>=0 then
                list[curr] = CurrLabel[m][n]
                curr = curr+1
                Count[CurrLabel[m][n]+2] = Count[CurrLabel[m][n]+2] + 1
                AllCount = AllCount+1
                --end
            end
        end
    end
    print(AllCount)
    Count= Count:mul(1/AllCount)
    return Count,list
end

