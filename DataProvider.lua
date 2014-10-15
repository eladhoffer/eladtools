local ffi = require 'ffi'
local lmdb = require 'lmdb'

local DataProvider = torch.class('DataProvider')

local function CatNumSize(num,size)
    local stg = torch.LongStorage(size:size()+1)
    stg[1] = num
    for i=2,stg:size() do
        stg[i]=size[i-1]
    end
    return stg
end
function DataProvider:__init(...)
    xlua.require('torch',true)

    local args = dok.unpack(
    {...},
    'InitializeData',
    'Initializes a DataProvider ',
    {arg='MaxNumItems', type='number', help='Number of Elements in each Batch',req = true},
    {arg='Name', type='string', help='Name of DataProvider',req = true},
    {arg='TensorType', type='string', help='Type of Tensor', default = 'torch.FloatTensor'},
    {arg='ExtractFunction', type='function', help='function used to extract Data, Label and Info', default= function(...) return ... end},
    {arg='Source', type='table', help='source of DataProvider', req=true},
    {arg='CachePrefix', type='string', help='path to caches data',default = '.'},
    {arg='CacheFiles', type='boolean', help='cache data into files', default=false},
    {arg='AutoLoad', type='boolean', help='load next data automaticaly', default=false},
    --{arg='DataContainer', type='boolean', help='true if size of Data == size of Items', default = true},
    {arg='CopyData', type='boolean', help='Copies data instead of referencing it ', default = true}
  
    )

    self.Name = args.Name
    self.MaxNumItems = args.MaxNumItems
    self.TensorType = args.TensorType
    self.ExtractFunction = args.ExtractFunction
    self.Source = args.Source
    self.CachePrefix = args.CachePrefix
    self.CacheFiles = args.CacheFiles
    --self.DataContainer = args.DataContainer
    self.AutoLoad = args.AutoLoad
    self.CopyData = args.CopyData

    if self.CacheFiles then
        os.execute('mkdir -p "' .. paths.dirname(self:BatchFilename(1)) .. '"')
    end
    
  
self.CurrentBatch = 0
    self.NumBatch = 0
  
    self.Data = torch.Tensor():type(self.TensorType)
    self.Labels = torch.Tensor():type(self.TensorType)
  self:Reset()
end


function DataProvider:size()
--if self.Items:dim() > 1 then
--    return self.Items:size(1)
--else return self.Items:size()[1]
--end
if self.Data:dim() == 0 then
    return 0
end
return self.Data:size(1)
end

function DataProvider:Reset()
      self.CurrentItemSource = 1
end
    

function DataProvider:__tostring__()
    local str = 'DataProvider:\n'
    if self:size() > 0 then
        str = str .. ' + num samples : '.. self:size()
    else
        str = str .. ' + empty set...'
    end
    return str
end

local function subdirs(path, listDirs )
    local listDirs = listDirs or {} --{paths.concat('.',path)}
    for f in paths.files(path) do
        local filename = paths.concat(path,f)
        if paths.dirp(filename) and (f~='..') and (f~='.') then
            table.insert(listDirs,filename)
        end
    end
    return listDirs
end


function DataProvider:BatchFilename(num) 
    return paths.concat(self.CachePrefix,self.Name .. '_Batch' .. num) 
end


function DataProvider:ShuffleItems()
    local RandOrder = torch.randperm(self.Data:size(1)):long()
    self.Data = self.Data:index(1,RandOrder)
    if self.Labels:dim() > 0 then
        self.Labels = self.Labels:index(1,RandOrder)
    end
    --print('(DataProvider)===>Shuffling Items')

end



function DataProvider:GetItems(location,num)
    --Assumes location and num are valid
    local num = num or 1
    local data
    local labels
    --if self.DataContainer then
        data = self.Data:narrow(1,location,num)
        labels = self.Labels:narrow(1,location,num) 
    --else
    --    local sizeData = self.Data:size()
    --    sizeData[1] = num
    --    local sizeLabels = self.Labels:size()
    --    sizeLabels[1] = num
    --    data = torch.Tensor(sizeData):typeAs(self.Data) 
    --    labels = torch.Tensor(sizeLabels):typeAs(self.Labels) 
    --    for i=1,num do
    --        data[i] = self.Data[self.Items[location+i-1]]
    --        labels[i] = self.Labels[self.Items[location+i-1]]
    --    end

    --end
    return data, labels
 end

function DataProvider:CurrentItemCount()
    return self.CurrentItemSource
end

function DataProvider:LoadBatch(batchnumber)
    local batchnumber = batchnumber or self.NumBatch
    local batchfilename = self:BatchFilename(batchnumber)
    if paths.filep(batchfilename) then
        print('(DataProvider)===>Loading Batch N.' .. batchnumber .. ' From ' .. batchfilename)
        local Batch = torch.load(batchfilename)
        self.Data = Batch.Data:type(self.TensorType)
        self.Labels = Batch.Labels:type(self.TensorType)
        self.NumBatch = batchnumber
        return true
    else
        return false
    end
end

function DataProvider:SaveBatch()
print('(DataProvider)===>Saving Batch')
    torch.save(self:BatchFilename(self.NumBatch), {Data = self.Data, Labels = self.Labels})
end

function DataProvider:CreateBatch()

    if self.CurrentItemSource > self.Source:size() then
        if not self.AutoLoad then
            return nil
        end

        if not self.Source:GetNextBatch() then
            return nil
        else
            self.CurrentItemSource = 1
        end
    end

    --print('(DataProvider)===>Creating Batch')
    local NumInBatch = math.min(self.Source:size() - self.CurrentItemSource + 1, self.MaxNumItems)
    local source_data, source_labels = self.Source:GetItems(self.CurrentItemSource, NumInBatch)
    local data, labels = self.ExtractFunction(source_data,source_labels)
    if self.CopyData then
        self.Data:resize(data:size())
        self.Data:copy(data)
        self.Labels:resize(labels:size())
        self.Labels:copy(labels)
    else
        self.Data = data
        self.Labels = labels
    end
    self.CurrentItemSource = self.CurrentItemSource + NumInBatch
    return true

end

function DataProvider:GetNextBatch()
    self.NumBatch = self.NumBatch + 1
    if self:LoadBatch() then
        return true
    elseif self:CreateBatch() then
        if self.CacheFiles then
            self:SaveBatch()
        end
        return true
    else
        return nil
    end
end













--------------------------File Searcher-------------------------------

function String2Tensor(string, lengthTensor)
    local x = torch.CharTensor(lengthTensor)
    local data=torch.data(x)        -- raw C pointer using torchffi
    ffi.copy(data, string)
return x
end

local FileSearcher, parent = torch.class('FileSearcher', 'DataProvider')
function FileSearcher:__init(...)
    local args = dok.unpack(
    {...},
    'InitializeData',
    'Initializes a DataProvider ',
    {arg='MaxNumItems', type='number', help='Number of Elements in each Batch', default = 1e8},
    {arg='Name', type='string', help='Name of DataProvider',req = true},
    {arg='ExtractFunction', type='function', help='function used to extract Data, Label and Info', default= function(...) return ... end},
    {arg='CachePrefix', type='string', help='path to caches data',default = '.'},
    {arg='CacheFiles', type='boolean', help='cache data into files', default=false},
    {arg='SubFolders', type='boolean', help='Recursive check for folders', default=false},
    {arg='PathList', type='table', help='Table of paths to search for files in', req = true},
    {arg='MaxFilenameLength', type='number', help='Maximum length of filename', default = 100}

    )


    self.maxStringLength = args.MaxFilenameLength
    parent:__init{
        MaxNumItems = args.MaxNumItems,
        Name = args.Name,
        TensorType = 'torch.CharTensor',
        CacheFiles = args.CacheFiles,
        CachePrefix = args.CachePrefix,
        Source = {}
    }
    self.NumBatch = 1
    if not self:LoadBatch() then
        

        local path = args.PathList
        local subfolders = args.SubFolders
        if subfolders then
            path = subdirs(path[1])
        end

        for i,p in pairs(path) do
            local num
 	    local numNewItems = tonumber(sys.execute('ls ' .. p .. '| wc -l')) -1
	    if i==1 then
           	self.Data = torch.CharTensor(numNewItems,self.maxStringLength)
		num = 1
	    else
		
            local currSize = self.Data:size()
                num = currSize[1] + 1
                currSize[1] = currSize[1] + numNewItems 
                self.Data:resize(currSize)
            end
            print('(DataProvider)===>Generating filenames from path' .. p)
            for f in paths.files(p) do
                local filename = paths.concat(p,f)
                if paths.filep(filename) then
		    if num <= self.Data:size(1) then
                    	self.Data[num] = String2Tensor(filename,self.maxStringLength)
                    	num = num+1
		    end
                end
            end
        end

      
        if self.CacheFiles then
           self:SaveBatch()
       end
    end
  --self:ShuffleItems()

end


function FileSearcher:GetItems(location, num)
    return self.Data:narrow(1,location,num)
--    local num = num or 1
--    local Filenames = torch.CharTensor(math.min(num,self:size(), self.Items:size(1)), self.Data:size(2))
--    for i=1,num do
--        Filenames[i] = self.Data[self.Items[i]]
--    end
--    return Filenames
end


function FileSearcher:GetNextBatch()
    return nil
end

-------------------------LMDB-----------------------------------
--local LMDB, parent = torch.class('LMDB', 'DataProvider')
--function LMDB:__init(...)
--  
--    parent:__init(...)
--self.env = lmdb.environment(self.CachePrefix .. self.Name,{subdir = false, max_dbs = 2})
--self.DataDB = env:db_open('Data')
--self.LabelsDB = env:db_open('Labels')

