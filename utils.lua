require 'image'

function PadTensor(im , SizeY, SizeX, pad, dy_middle, dx_middle)
    local dx_middle = dx_middle or 0
    local dy_middle = dy_middle or 0
    local pad = pad or 0
    if im:dim()>2 then
        local padded = torch.zeros(im:size(1), SizeY,SizeX):fill(pad):typeAs(im)
        for i=1, im:size(1) do
            padded[i]:copy(PadTensor(im[i], SizeY, SizeX, pad, dy_middle, dx_middle))
        end
        return padded
    else
        if (im:size(1) == SizeY) and (im:size(2) == SizeX) then
            return im
        end
        if (im:size(1) > SizeY) or (im:size(2) > SizeX) then
            return PadTensor(im:narrow(1, 1,math.min(SizeY, im:size(1))):narrow(2, 1,math.min(SizeX, im:size(2))), SizeY, SizeX, pad)
        end
        local padded = torch.zeros(SizeY,SizeX):fill(pad):typeAs(im)
        local hpad = math.floor((SizeY - im:size(1))/2)
        local wpad = math.floor((SizeX - im:size(2))/2)
        local starty = hpad+1
        local endy = im:size(1) + hpad
        local startx = wpad+1
        local endx = im:size(2) + wpad
        padded[{{starty+dy_middle,endy+dy_middle},{startx+dx_middle,endx+dx_middle}}]:copy(im)
        return padded
    end
end


function CropCenter(Data, side)
    if (Data:dim() == 2) then
        local resized = CropCenter(Data:reshape(1,Data:size(1), Data:size(2)), side)
        return resized:reshape(resized:size(2),resized:size(3))
    end
    local x1 = math.max(math.floor((Data:size(3) - side) / 2),1)
    local y1 = math.max(math.floor((Data:size(2) - side) / 2),1)
    local x2 = math.min(x1 + side -1,Data:size(3))
    local y2 = math.min(y1 + side -1,Data:size(2))
    local CroppedData = Data[{{},{y1,y2},{x1,x2}}]
    return CroppedData
end

rnd_rotate = function(angle_range)

    apply_rot = function(Data)
        angle = math.random(-angle_range,angle_range)
        mirrored = math.random(0,1)==1
        Data = image.rotate(Data,math.rad(angle))

        if (mirrored) then
            Data = image.hflip(Data)
        end
        return Data
    end
    return apply_rot
end	


rnd_rotate_crop = function(angle_range,side)

    apply_rot = function(Data)
        angle = math.random(-angle_range,angle_range)
        mirrored = math.random(0,1)==1
        if (Data:size()[2] ~= side) then --don't rotate and crop if already same size
            Data = image.rotate(Data,math.rad(angle))
            Data = crop_center(Data,side)
        end
        if (mirrored) then
            Data = image.hflip(Data)
        end

        return Data
    end
    return apply_rot
end	


