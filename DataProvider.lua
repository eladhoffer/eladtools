local DataProvider = torch.class('util.DataProvider')

function DataProvider:__init(...)
   xlua.require('image',true)
   self.numSamples = 0
   if select('#',...) > 0 then
  	self:load(...)
   end

   self.BatchFilename = function(num) return paths.concat(self.CachePrefix,num) end
end

function DataProvider:size()
   return self.numSamples
end

function DataProvider:__tostring__()
   str = 'DataProvider:\n'
   if self.numSamples then
  	str = str .. ' + num samples : '..self.numSamples
   else
  	str = str .. ' + empty set...'
   end
   return str
end

function DataProvider:load(...)
   -- parse args
   local args, dataSetFolder, numSamplesRequired, cacheFile, channels,
   sampleSize,padding
  	= xlua.unpack(
  	{...},
  	'DataProvider.load', nil,
  	{arg='dataSetFolder', type='string', help='path to dataset', req=true},
  	{arg='numSamplesRequired', type='number', help='number of patches to load', default='all'},
  	{arg='cachePrefix', type='string', help='path to file to cache files'},
  	{arg='sampleSize', type='table', help='resize all sample: {c,w,h}'},
  	{arg='padding', type='boolean', help='center sample in w,h dont rescale'}
   )
   self.CachePrefix = CachePrefix or './'
   self.cacheFileName = c or self.cacheFileName
self.CurrentItemOverall
self.CurrentItemBatch
self.BatchSize





function DataProvider:GenerateFilenames(path, subfolders)
    local subfolders = subfolders or false
    if type(path)=='table' then
        for _,p in pairs(path) do
            self:GenerateFilenames(p,subfolders)
        end
    else
        for f in paths.files(path) do
            local filename = paths.concat(path,f)
            if paths.dirp(filename) and subfolders and (f~='..') and (f~='.') then
                self:GenerateFilenames(filename, true)
            elseif paths.filep(filename) then
                table.insert(self.Filenames, filename)
            end
        end
    end
end

function DataProvider:AddPrepFunc(prep)
    table.insert(self.Preprocessors, prep)
    local NewItems = torch.LongTensor(#self.Filenames, self.NumClassLabels + 2)
    NewItems[{{},1}] = torch.range(1,#self.Filenames)
    NewItems[{{},self.NumClassLabels+2}]:fill(#self.Preprocessors)
    NewItems[{{},{2,self.NumClassLabels+1}}] = self.LabelFunc(self.Filenames)

    self.Items = self.Items:cat(NewItems,1)
end

local function CreateData()
    local Attributes = {}
    local curr = #Filenames+1
    for i=1, m:size(2) do
        local ClassID = tostring(m[1][i])
        local NumImg = tostring(m[2][i])
        if #ClassID < 8 then
            ClassID = "0" .. ClassID
        end
        ClassID = "n" .. ClassID
        local filename  = path .. ClassID .. "/" .. ClassID .. '_' .. NumImg .. ".JPEG"
        --print(filename)

        if paths.filep(filename) then
            --print('Valid')
            Filenames[curr] = filename
            Attributes[curr] = m[{{3,27},i}]
            curr = curr+1
        end
    end
    return Filenames, tableToTensor(Attributes)
end

function DataProvider:InitBatch(numElements,typeTensor)
    local BatchSize = {NumElements}
    for _,k in pairs(self.sampleSize) do
        table.insert(BatchSize,k)
    end
    self.Batch = torch.Tensor(torch.LongStorage(BatchSize))
    if typeTensor then
        self.Batch = self.Batch:type(typeTensor)
    end
    return self.Batch
end

function DataProvider:InitMiniBatch(numElements,typeTensor)
    local BatchSize = {NumElements}
    for _,k in pairs(self.sampleSize) do
        table.insert(BatchSize,k)
    end
    self.MiniBatch = torch.Tensor(torch.LongStorage(BatchSize))
    if typeTensor then
        self.MiniBatch = self.MiniBatch:type(typeTensor)
    end
    return self.MiniBatch
end

function DataProvider:loadBatch(batchnumber)
    local batchnumber = batchnumber or self.CurrentBatchNum
    if paths.filep(DataProvider.BatchFilename(batchnumber)) then
        self.Batch = torch.load(paths.concat(CachePrefix,filename))
        self.CurrentBatchNum = batchnumber
        return true
    else
        return false
    end
end

function DataProvider:saveBatch()
    self.Batch = torch.save(DataProvider.BatchFilename(self.CurrentBatchNum), self.Batch)
end

function DataProvider:CreateBatch()
    if self.Items:size(1) < self.CurrentItemOverall then
        return false
    end
    
    if self.Items:size(1) < self.CurrentItemOverall + self.Batch:size(1) - 1 then
        self:InitBatch(self.Items:size(1) - self.CurrentItemOverall + 1)
    end
    for i = 1,self.Batch:size(1) do
        local PrepFunc = self.Preprocessors[self.Items[i][#NumClassLabels+1]]
        local Sample = self.FileLoader(self.Filenames[self.Items[i][1]])
        self.Batch[i] = PrepFunc(Sample)
        xlua.progress(i, self.Batch:size(1))
        self.CurrentItemOverall = self.CurrentItemOverall + 1
    end
    self.CurrentBatchNum = self.CurrentBatchNum + 1
    return self.Batch
end


function DataProvider:GetNextBatch()
    if self.NumBatches < self.CurrentBatchNum then
        self.CurrentBatchNum = self.CurrentBatchNum + 1
        if not self:loadBatch() then
            DataProvider:CreateBatch()
        end
        return true
    else
        return false
    end
end
