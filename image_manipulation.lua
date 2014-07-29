require 'image'

function PadTensor(im , SizeY, SizeX, pad, dy_middle, dx_middle)
    local dx_middle = dx_middle or 0
    local dy_middle = dy_middle or 0
    local pad = pad or 0
    if im:dim()>2 then

        local padded = torch.zeros(im:size(1), SizeY,SizeX):fill(pad):typeAs(im)
        for i=1, im:dim() do
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


crop_center = function(Data, side)
	if (Data:dim() == 2) then
        return crop_center(Data:reshape(1,Data:size(1), Data:size(2)), side)
    end
	local x1 = math.max(math.floor((Data:size(3) - side) / 2),1)
    local y1 = math.max(math.floor((Data:size(2) - side) / 2),1)
    local x2 = math.min(x1 + side -1,Data:size(3))
    local y2 = math.min(y1 + side -1,Data:size(2))
    --Data = image.crop(Data,x1,y1,x2,y2)
    Data = Data[{{},{y1,y2},{x1,x2}}]
	return Data
end

crop_9_block = function(Data, side, block_num)
    local x1,y1
    if (block_num==1) then
        x1 = 1
        y1 = 1
    elseif (block_num==2) then
        y1 = 1
        x1 = math.max(math.floor((Data:size(3) - side) / 2),1)
    elseif (block_num==3) then
        y1 = 1
        x1 = math.max(Data:size(1)-side, 1)
    elseif (block_num==4) then
        x1 = 1
        y1 = math.max(math.floor((Data:size(2) - side) / 2),1)
    elseif (block_num==5) then
        x1 = math.max(math.floor((Data:size(3) - side) / 2),1)
        y1 = math.max(math.floor((Data:size(2) - side) / 2),1)
    elseif (block_num==6) then
        x1 = math.max(Data:size(1)-side, 1)
        y1 = math.max(math.floor((Data:size(2) - side) / 2),1)
    elseif (block_num==7) then
        y1 = math.max(Data:size(2)-side, 1)
        x1 = 1
    elseif (block_num==8) then
        y1 = math.max(Data:size(2)-side, 1)
        x1 = math.max(math.floor((Data:size(3) - side) / 2),1)
    elseif (block_num==9) then
        x1 = math.max(Data:size(1)-side, 1)
        y1 = math.max(Data:size(2)-side, 1)
    else
        print ('Block number must be 1-9')
        return
    end

    local x2 = math.min(x1 + side,Data:size(3))
    local y2 = math.min(y1 + side,Data:size(2))
    Data = image.crop(Data,x1,y1,x2,y2)
    if (Data:size()[2]<side) or (Data:size()[3] < side) then
		Data = image.scale(Data, side, side, bilinear)
	end
	return Data
end

crop_and_scale = function(crop_size,scale_size)
    CnS_func = function(Data)
        return(image.scale(crop_center(Data,crop_size),scale_size,scale_size,bilinear))
    end
    return CnS_func
end

crop_center_64 = function(Data)
	return crop_center(Data,64)
end

crop_center_100 = function(Data)
	return crop_center(Data,100)
end

crop_center_160 = function(Data)
	return crop_center(Data,160)
end

scale_center_32 = function(Data)
	
	return image.scale(crop_center_64(Data),32,32,bilinear)

	--return crop_center(Data,32)
end

adapt_lfw = function(Data)
	Data = crop_center_100(Data)
	return image.scale(crop_center_64(Data),32,32,bilinear)

end


resize = function(side)
	apply_resize = function(Data)
		return image.scale(Data, side, side, bilinear)
	end
	return apply_resize;
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


crop_rnd_block = function(size)
    f = function(Data)
        local block_num = math.random(1,9)
        return crop_9_block(Data, size, block_num)
    end
    return f
end

