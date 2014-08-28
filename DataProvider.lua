require 'xlua'

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
    if #self.Items > 0 then
        str = str .. ' + num samples : '.. #self.Items
    else
        str = str .. ' + empty set...'
    end
    return str
end

function DataProvider:load(CachePrefix, sampleSize, labelSize, CacheFiles)
    -- parse args
    self.CachePrefix = CachePrefix or '.'
    self.CacheFiles = CacheFiles or false
    self.sampleSize = sampleSize
    self.labelSize = labelSize or {1}
    self.Preprocessors = {}
    if #self.labelSize == 1 then
        self.labelSize = self.labelSize[1]
    end
    self.itemslistFile = paths.concat(self.CachePrefix, 'ItemsList')
    if paths.filep(self.itemslistFile) then
        self.Items = torch.load(self.itemslistFile)
        self.TotalnumItems = #self.Items
    else
        os.execute('mkdir -p "' .. paths.dirname(self.itemslistFile) .. '"')
        self.Items = {}
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
    return #self.Items > 0
end

function DataProvider:ShuffleItems()
    local n = #self.Items
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
        print('(DataProvider)===>Generating filenames from path' .. p)
        for f in paths.files(p) do
            local filename = paths.concat(p,f)
            if paths.filep(filename) then
                table.insert(self.Items, {Filename = filename, Label = {}, Transformation = 1})
                self.TotalnumItems = self.TotalnumItems + 1
            end
        end
    end

end

function DataProvider:AddFilenames(fs, searchpaths)
    local path = searchpaths or {'.'}

    if type(path) =='table' then
        for _,p in pairs(path) do
            self:AddFilenames(fs,p)
        end
    else
        for _,f in pairs(fs) do
            local filename = paths.concat(path,f)
            if paths.filep(filename) then
                table.insert(self.Items, {Filename = filename, Label = {}, Transformation = 1})
                self.TotalnumItems = self.TotalnumItems + 1
            end
        end
    end
end

function DataProvider:LabelItems(LabelFunc)
    for _,item in pairs(self.Items) do
        item.Label = LabelFunc(item)
    end
end

function DataProvider:AddPrepFunc(prep)
    table.insert(self.Preprocessors, prep)
    if #self.Preprocessors > 1 then
        local numItems = #self.Items
        for i=1, numItems do
            table.insert(self.Items, {Filename = self.Items[i].Filename, Label = self.Items[i].Label, Transformation = #self.Preprocessors})
        end
    end
end

function DataProvider:InitBatch(numElements,typeTensor)
    local numElements = numElements or #self.Items
    self.NumBatchElements = numElements
    local BatchSize = {numElements}
    local BatchLabelsSize = {numElements}
    for _,k in pairs(self.sampleSize) do
        table.insert(BatchSize,k)
    end
    if type(self.lableSize) == 'table' then
        for _,k in pairs(self.lableSize) do
            table.insert(BatchLabelsSize,k)
        end
    else 
        if self.labelSize > 1 then
            table.insert(BatchLabelsSize,self.labelSize)
        end
    end
    self.Batch = {}
    self.Batch.Data = torch.Tensor(torch.LongStorage(BatchSize))
    self.Batch.Labels = torch.Tensor(torch.LongStorage(BatchLabelsSize))
    if typeTensor then
        self.Batch.Data = self.Batch.Data:type(typeTensor)
        self.Batch.Labels = self.Batch.Labels:type(typeTensor)
    end
    self.CurrentBatchNum = 0
    self.NumBatches = math.ceil(self.TotalnumItems/self.NumBatchElements)
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

function DataProvider:InitMiniBatch(numElements,typeTensor)
    self.NumMiniBatchElements = numElements
    local BatchSize = {numElements}
    local BatchLabelsSize = {numElements}
    for _,k in pairs(self.sampleSize) do
        table.insert(BatchSize,k)
    end
    if type(self.lableSize) == 'table' then
        for _,k in pairs(self.lableSize) do
            table.insert(BatchLabelsSize,k)
        end
    else 
        if self.labelSize > 1 then
            table.insert(BatchLabelsSize,self.labelSize)
        end
    end
    self.MiniBatch = {}
    self.MiniBatch.Data = torch.Tensor(torch.LongStorage(BatchSize))
    self.MiniBatch.Labels = torch.Tensor(torch.LongStorage(BatchLabelsSize))
    if typeTensor then
        self.MiniBatch.Data = self.MiniBatch.Data:type(typeTensor)
        self.MiniBatch.Labels = self.MiniBatch.Labels:type(typeTensor)
    end
    self.CurrentItemBatch = 1
    return self.MiniBatch
end

function DataProvider:LoadBatch(batchnumber)
    local batchnumber = batchnumber or self.CurrentBatchNum
    local batchfilename = self:BatchFilename(batchnumber)
    if paths.filep(batchfilename) then
        print('(DataProvider)===>Loading Batch N.' .. batchnumber .. ' From ' .. batchfilename)
        self.Batch = torch.load(batchfilename)
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
    if #self.Items == 0 then
        return nil
    end

    if #self.Items <  self.NumBatchElements then
        self:InitBatch(#self.Items )
    end
    print('(DataProvider)===>Creating Batch')
    for i = 1,self.NumBatchElements do
        local Item = table.remove(self.Items,1)
        local PrepFunc = self.Preprocessors[Item.Transformation]
        local Sample = self.FileLoader(Item.Filename)
        self.Batch.Data[i] = PrepFunc(Sample)
        self.Batch.Labels[i] = Item.Label
        xlua.progress(i, self.NumBatchElements)
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
        return true
    else
        self.StopLoading = true
        return nil
    end
end

function DataProvider:GetNextMiniBatch(autoLoadBatch)
    local autoLoadBatch = autoLoadBatch or false
    if self.StopLoading then
        return nil
    end
    if self.CurrentItemBatch + self.NumMiniBatchElements > self.NumBatchElements or self.CurrentBatchNum == 0 then
        if (not autoLoadBatch) then
            return nil
        end
        if (not self:GetNextBatch()) then
            self.StopLoading = true
            return nil
        end
    end

    self.MiniBatch.Data:copy(self.Batch.Data:narrow(1,self.CurrentItemBatch, self.NumMiniBatchElements))
    self.MiniBatch.Labels:copy(self.Batch.Labels:narrow(1,self.CurrentItemBatch, self.NumMiniBatchElements))
    self.CurrentItemBatch = self.CurrentItemBatch + self.NumMiniBatchElements
    self.CurrentItem = self.CurrentItem + self.NumMiniBatchElements
    --local b = self.MiniBatch.Data
    --local B = self.Batch.Data
    --for i=1,self.NumMiniBatchElements do
    --    --if self.CurrentItemBatch > self.NumBatchElements or self.CurrentBatchNum == 0 then
    --    --    if (not autoLoadBatch) or (not self:GetNextBatch()) then
    --    --        self.StopLoading = true
    --    --        return nil
    --    --    end
    --    --end
    --    b[i] = B[self.CurrentItemBatch]
    --    self.MiniBatch.Labels[i] = self.Batch.Labels[self.CurrentItemBatch]
    --    self.CurrentItemBatch = self.CurrentItemBatch + 1
    --    self.CurrentItem = self.CurrentItem + 1
    --end
    return true
end



