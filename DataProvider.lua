require 'xlua'
local ffi = require 'ffi'

local maxStringLength = 60 -- this has to be predetermined
local DataProvider = torch.class('DataProvider')
function DataProvider:__init(...)
    xlua.require('image',true)
    xlua.require('torch',true)
    self:load(...)
end

function DataProvider:size()
    return self.TotalnumItems
end

function DataProvider:__tostring__()
    str = 'DataProvider:\n'
    if self.TotalnumItems > 0 then
        str = str .. ' + num samples : '.. self.TotalnumItems
    else
        str = str .. ' + empty set...'
    end
    return str
end
--[[numStrings = 10                    -- for example, lets do 10, but this number can be anything upto memory limits
maxStringLength = 100          -- this has to be predetermined
 
-- allocate CharTensor
bigStringTensor = torch.CharTensor(numStrings, maxStringLength)
bst_data=torch.data(bigStringTensor)        -- raw C pointer using torchffi
 
-- load some strings into the stringTensor
str='hello world'
 
-- write strings to tensor
for i=1,bigStringTensor:size(1) do
      local stri = str .. ' ' .. i
      ffi.copy(bst_data + ((i-1) * maxStringLength), stri)
end
 
-- read back data gotcha (torchffi seems to always return the parent tensor's data pointer)
for i=1,bigStringTensor:size(1) do
      print(ffi.string(torch.data(bigStringTensor[i])))
end
 
-- read back properly
for i=1,bigStringTensor:size(1) do
      print(ffi.string(bst_data + ((i-1) * maxStringLength)))
      end]]--
function DataProvider:load(...)
    -- parse args
    local args = dok.unpack(
    {...},
    'DataProvider:load',
    'Loads a DataProvider',
    {arg='CachePrefix', type='string', help='path to caches data',default = '.'},
    {arg='CacheFiles', type='boolean', help='cache data into files', default=false},
    {arg='PreAllocItems', type='number', help='Number of Items', default=-1}
    )
    self.CachePrefix = args.CachePrefix
    self.CacheFiles = args.CacheFiles
    self.itemslistFile = paths.concat(self.CachePrefix, 'ItemsList')
    if paths.filep(self.itemslistFile) then
        self.Items = torch.load(self.itemslistFile)
        self.TotalnumItems= self.Items.Filenames:size(1)
    else
        os.execute('mkdir -p "' .. paths.dirname(self.itemslistFile) .. '"')
        self.Items = {}
        self.Items.Filenames = torch.CharTensor(1,maxStringLength)
        self.TotalnumItems = 0
    end

    self.CurrentItem = 1
    self.FileLoader = image.loadJPG
end
function subdirs(path, listDirs )
    local listDirs = listDirs or {paths.concat('.',path)}
    for f in paths.files(path) do
        local filename = paths.concat(path,f)
        if paths.dirp(filename) and (f~='..') and (f~='.') then
            table.insert(listDirs,filename)
        end
    end
    return listDirs
end

function DataProvider:BatchFilename(num) 
    return paths.concat(self.CachePrefix,'Batch' .. num) 
end

function DataProvider:SaveItemsList()
    torch.save(self.itemslistFile, self.Items)
end

function DataProvider:ItemsLoaded()
    return self.TotalnumItems > 0
end

function DataProvider:ShuffleItems()
    local n = self.TotalnumItems
    while n > 2 do
        local k = math.random(n)
        self.Items[n], self.Items[k] = self.Items[k], self.Items[n]
        n = n - 1
    end
end

function DataProvider:ShuffleBatch()
    local n = self.Batch.Data:size(1)
    while n > 2 do
        local k = math.random(n)
        self.Batch.Data[n], self.Batch.Data[k] = self.Batch.Data[k]:clone(), self.Batch.Data[n]:clone()
        self.Batch.Labels[n], self.Batch.Labels[k] = self.Batch.Labels[k], self.Batch.Labels[n]
        n = n - 1
    end
end

function DataProvider:InsertItem(filename)
    if self.Items.Filenames:size(1) > self.TotalnumItems then
        self.TotalnumItems = self.TotalnumItems + 1
        local Filenames = self.Items.Filenames
        local data=torch.data(Filenames)        -- raw C pointer using torchffi
        ffi.copy(data + ((self.TotalnumItems-1) * maxStringLength), filename)
    end
end

function DataProvider:GetItem(location)
    if self.Items.Filenames:size(1) < location then
        return nil
    end
    local Filenames = self.Items.Filenames
    local data=torch.data(Filenames)        -- raw C pointer using torchffi
    return ffi.string(data + ((location-1) * maxStringLength))
end


function DataProvider:GenerateFilenames(path, subfolders)
    local subfolders = subfolders or false
    if subfolders then
        path = subdirs(path)
    end
    local path_list
    if type(path)=='table' then
        path_list = path
    else
        path_list = {path}
    end
    for _,p in pairs(path_list) do
        
        local currSize = self.Items.Filenames:size()
        local numNewItems = tonumber(sys.execute('ls ' .. p .. '| wc -l')) -1
        if currSize[1] - self.TotalnumItems < numNewItems then
            currSize[1] = currSize[1] + numNewItems
            self.Items.Filenames:resize(currSize)
        end
        print('(DataProvider)===>Generating filenames from path' .. p)
        for f in paths.files(p) do
            local filename = paths.concat(p,f)
            if paths.filep(filename) then
                self:InsertItem(filename)
            end
        end
    end

end
--
--function DataProvider:PushItems(items)
--    local n=items:
--    for i=1,n do
--        table.insert(self.Items, table.remove(items,1))
--        self.TotalnumItems = self.TotalnumItems + 1
--    end
--end
--
--function DataProvider:PopItems(num)
--    local removedItems = {}
--    local n = self.TotalnumItems
--    for i=1,math.min(num,n) do
--        table.insert(removedItems, table.remove(self.Items,1))
--        self.TotalnumItems = self.TotalnumItems - 1
--    end
--    return removedItems
--end

function DataProvider:AddFilenames(fs, searchpaths)
    local path = searchpaths or {'.'}

    if type(path) =='table' then
        for _,p in pairs(path) do
            self:AddFilenames(fs,p)
        end
    else
        local currSize = self.Filenames:size()
        local numNewItems = tonumber(sys.execute('ls ' .. fs .. '| wc -l')) -2
        print(numNewItems)
        if curSize[1] - self.TotalnumItems < numNewItems then
            currSize[1] = currSize[1] + numNewItems
            self.Items.Filenames:resize(currSize)
        end
        for _,f in pairs(fs) do
            local filename = paths.concat(path,f)
            if paths.filep(filename) then
                self:InsertItem(filename)
            end
        end
    end
end

function DataProvider:LabelItems(LabelFunc)
    local FirstLabel = LabelFunc(self:GetItem(1))
    if torch.type(FirstLabel) == 'number' then
        self.Items.Labels = torch.Tensor(self.TotalnumItems)
        -- else
        --   local

    end
    for i=1,self.TotalnumItems do
        self.Items.Labels[i] = LabelFunc(self:GetItem(i))
    end
end

local function InitializeData(...)
local args = dok.unpack(
    {...},
    'InitializeData',
    'Initializes a DataProvider Batch or MiniBatch',
    {arg='NumElements', type='number', help='Number of Elements in each Batch',req = true},
    {arg='SampleSize', type='table', help='Size of each sample in Batch', req=true},
    {arg='LabelSize', type='table', help='Size of each label in Batch', default = {1}},
    {arg='TensorType', type='string', help='Type of Tensor', default = nil},
    {arg='ExtractFunction', type='Function', help='function used to extract Data and label', default=nil}
    )

    local numElements = args.NumElements
    local BatchSize = {numElements}
    local BatchLabelsSize = {numElements}
    for _,k in pairs(args.SampleSize) do
        table.insert(BatchSize,k)
    end
    if #args.LabelSize >1 then
        for k=1,#args.LabelSize do
            table.insert(BatchLabelsSize,args.LabelSize[k])
        end
    else 
        if args.LabelSize[1] > 1 then
            table.insert(BatchLabelsSize,args.LabelSize[1])
        end
    end
    Batch = {}
    Batch.Data = torch.Tensor(torch.LongStorage(BatchSize))
    Batch.Labels = torch.Tensor(torch.LongStorage(BatchLabelsSize))
    Batch.MaxNumElements = numElements
    Batch.ExtractFunction = args.ExtractFunction
    if args.TensorType then
        Batch.Data = Batch.Data:type(args.TensorType)
        Batch.Labels = Batch.Labels:type(args.TensorType)
    end
    return Batch
end


function DataProvider:InitBatch(...)
    self.Batch = InitializeData(...)
    self.CurrentBatchNum = 0
    self.NumBatches = math.ceil(self.TotalnumItems/self.Batch.Data:size(1))
    return self.Batch
end

function DataProvider:InitBatchData(data)
    self.Batch.Data = data.Labels
    self.Batch.Labels = data.Labels
end

function DataProvider:ResetCount(batch)
    self.CurrentBatchNum = batch or 0
    self.CurrentItemBatch = 1
    self.CurrentItem = 1
    self.StopLoading = false

end

function DataProvider:CurrentItemCount()
    return self.CurrentItem
end

function DataProvider:InitMiniBatch(...)
    self.MiniBatch = InitializeData(...)
    self.CurrentItemBatch = 1
    return self.MiniBatch
end

function DataProvider:LoadBatch(batchnumber)
    local batchnumber = batchnumber or self.CurrentBatchNum
    local batchfilename = self:BatchFilename(batchnumber)
    if paths.filep(batchfilename) then
        print('(DataProvider)===>Loading Batch N.' .. batchnumber .. ' From ' .. batchfilename)
        local TensorType = self.Batch.Data:type()
        self.Batch = torch.load(batchfilename)
        self.Batch.Data = self.Batch.Data:type(TensorType)
        self.Batch.Labels = self.Batch.Labels:type(TensorType)
        self.CurrentBatchNum = batchnumber
        return true
    else
        return false
    end
end

function DataProvider:SaveBatch()
    torch.save(self:BatchFilename(self.CurrentBatchNum), self.Batch)
end

function DataProvider:CreateBatch()
    if self.TotalnumItems == 0 then
        return nil
    end
    if self.TotalnumItems <  self.Batch.Data:size(1) then
        self.Batch.Data = self.Batch.Data:narrow(1,1,self.TotalnumItems)
        self.Batch.Labels = self.Batch.Labels:narrow(1,1,self.TotalnumItems)
    else 
        if self.Batch.Data:size(1) < self.Batch.MaxNumElements then
            local BatchDataSize = self.Batch.Data:size()
            local BatchLabelSize= self.Batch.Label:size()
            BatchDataSize[1] = self.Batch.NumMaxElements
            BatchLabelSize[1] = self.Batch.NumMaxElements
            self.Batch.Data:resize(BatchDataSize)
            self.Batch.Labels:resize(BatchLabelSize)
        end
    end
   print('(DataProvider)===>Creating Batch')
    for i = 1,self.Batch.Data:size(1) do
        local PrepFunc = self.Batch.ExtractFunction
        local Sample = self.FileLoader(self:GetItem(i))
        self.Batch.Data[i] = PrepFunc(Sample)
        self.Batch.Labels[i] = self.Items.Labels[i]
        xlua.progress(i, self.Batch.Data:size(1))
    end
    return self.Batch
end


function DataProvider:GetNextBatch()
    if (self.NumBatches == 1) and (self.CurrentBatchNum == 1) then
        self:ResetCount(1)
        return true
    end
    if self.StopLoading then
        return nil
    end
    if self.NumBatches > self.CurrentBatchNum then
        self.CurrentBatchNum = self.CurrentBatchNum + 1
        if not self:LoadBatch() then
            self:CreateBatch()
            if self.CacheFiles then
                self:SaveBatch()
            end
        end
        self.CurrentItemBatch = 1
        if self.NumBatches == self.CurrentBatchNum then
            self.StopLoading = true
        end
        return true
    else
        self.StopLoading = true
        return nil
    end
end

function DataProvider:EndOfData()
    return self.StopLoading
end

function DataProvider:GetNextMiniBatch(autoLoadBatch)
    local autoLoadBatch = autoLoadBatch or false
    if self.CurrentItemBatch + self.MiniBatch.Data:size(1) > self.Batch.Data:size(1) or self.CurrentBatchNum == 0 then
        if (not autoLoadBatch) then
            return nil
        end
        if (not self:GetNextBatch()) then
            return nil
        end
    end
    local Data,Labels
    if self.MiniBatch.ExtractFunction then
        Data, Labels = self.MiniBatch.ExtractFunction(self.Batch.Data:narrow(1,self.CurrentItemBatch, self.MiniBatch.Data:size(1)), 
        self.Batch.Labels:narrow(1,self.CurrentItemBatch, self.MiniBatch.Data:size(1)))
    else
        Data = self.Batch.Data:narrow(1,self.CurrentItemBatch, self.MiniBatch.Data:size(1))
        Labels = self.Batch.Labels:narrow(1,self.CurrentItemBatch, self.MiniBatch.Data:size(1))
    end

    --if self.MiniBatch.Data:type() == Data:type() then
    --    self.MiniBatch = Data
    self.MiniBatch.Data:copy(Data)
    self.MiniBatch.Labels:copy(Labels)

    self.CurrentItemBatch = self.CurrentItemBatch + self.MiniBatch.Data:size(1)
    self.CurrentItem = self.CurrentItem + self.MiniBatch.Data:size(1)

    return true
end



