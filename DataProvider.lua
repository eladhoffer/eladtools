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

function DataProvider:load(CachePrefix, sampleSize, labelSize)
    -- parse args
    self.CachePrefix = CachePrefix or '.'
    self.sampleSize = sampleSize
    self.labelSize = labelSize or {1}
    if #self.labelSize == 1 then
        self.labelSize = self.labelSize[1]
    end
    self.itemslistFile = paths.concat(self.CachePrefix, 'ItemsList')
    if paths.filep(self.itemslistFile) then
        self.Items = torch.load(self.itemslistFile)
        self.TotalnumItems = #self.Items
    else
        self.Items = {}
        self.TotalnumItems = 0
    end

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
    return paths.concat(self.CachePrefix,num) 
end

function DataProvider:SaveItemsList()
    torch.save(self.itemslistFile, self.Items)
end

function DataProvider:ItemsLoaded()
    return #self.Items > 0
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
        print('Generating filenames from path' .. p)
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


function DataProvider:AddPrepFunc(prep)
    table.insert(self.Preprocessors, prep)
    local numItems = #self.Items
    for i=1, numItems do
        table.insert(self.Items, {Filename = self.Items[i].Filename, Label = self.Items[i].Label, Transformation = #self.Preprocessors})
    end
end

function DataProvider:InitBatch(numElements,typeTensor)
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
        if self.sampleSize > 1 then
            table.insert(BatchLabelsSize,self.sampleSize)
        end
    end
    self.Batch.Data = torch.Tensor(torch.LongStorage(BatchSize))
    self.Batch.Labels = torch.Tensor(torch.LongStorage(BatchLabelsSize))
    if typeTensor then
        self.Batch.Data = self.Batch.Data:type(typeTensor)
        self.Batch.Labels = self.Batch.Labels:type(typeTensor)
    end
    return self.Batch
end

function DataProvider:InitBatchData(data)
    self.Batch.Data = data.Labels
    self.Batch.Labels = data.Labels
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
        if self.sampleSize > 1 then
            table.insert(BatchLabelsSize,self.sampleSize)
        end
    end
    self.MiniBatch.Data = torch.Tensor(torch.LongStorage(BatchSize))
    self.MiniBatch.Labels = torch.Tensor(torch.LongStorage(BatchLabelsSize))
    if typeTensor then
        self.MiniBatch.Data = self.Batch.Data:type(typeTensor)
        self.MiniBatch.Labels = self.Batch.Labels:type(typeTensor)
    end
    return self.MiniBatch
end

function DataProvider:loadBatch(batchnumber)
    local batchnumber = batchnumber or self.CurrentBatchNum
    local batchfilename = self:BatchFilename(batchnumber)
    if paths.filep(batchfilename) then
        self.Batch = torch.load(batchfilename)
        self.CurrentBatchNum = batchnumber
        return true
    else
        return false
    end
end

function DataProvider:saveBatch()
    torch.save(DataProvider.BatchFilename(self.CurrentBatchNum), self.Batch)
end

function DataProvider:CreateBatch()
    if #self.Items == 0 then
        return nil
    end

    if #self.Items <  self.NumBatchElements then
        self:InitBatch(#self.Items )
    end
    for i = 1,self.NumBatchElements do
        local PrepFunc = self.Preprocessors[self.Items[1].Transformation]
        local Sample = self.FileLoader(self.Items[1].Filename)
        local Label = self.LabelFunction(self.Item[1])
        self.Batch.Data[i] = PrepFunc(Sample)
        self.Bath.Labels[i] = Label
        xlua.progress(i, self.NumBatchElements)
        table.remove(self.Items,1)
    end
    self.CurrentBatchNum = self.CurrentBatchNum + 1
    return self.Batch
end


function DataProvider:GetNextBatch()
    if self.EndOfData then
        return nil
    end
    if self.NumBatches < self.CurrentBatchNum then
        self.CurrentBatchNum = self.CurrentBatchNum + 1
        if not self:loadBatch() then
            DataProvider:CreateBatch()
        end
        self.CurrentItemBatch = 1
        return self.Batch
    else
        return nil
    end
end

function DataProvider:GetNextMiniBatch()
    if self.EndOfData then
        return nil
    end

    for i=1,self.NumMiniBatchElements do
        if self.CurrentItemBatch > self.NumBatchElements then
            if not self:GetNextBatch() then
                self.EndOfData = true
                return self.MiniBatch.Data, self.MiniBatch.Labels
            end
        end
        self.MiniBatch[i].Data = self.Batch.Data[self.CurrentItemBatch]
        self.MiniBatch[i].Labels = self.Batch.Labels[self.CurrentItemBatch]
        self.CurrentItemBatch = self.CurrentItemBatch + 1
    end
    return self.MiniBatch.Data, self.MiniBatch.Labels
end



