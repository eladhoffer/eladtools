require 'DataProvider'

data = DataProvider('/home/ehoffer/Datasets/Cache', {3,256,256})
if not data:ItemsLoaded() then
data:GenerateFilenames('/home/ehoffer/Datasets/ImageNet/Attributes/',true)
end
