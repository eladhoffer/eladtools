require 'Dataset'
d= torch.load('./RandomDataSet')

--l_train5 = CreateFullList(d,1,1)
l_train5 = CreateRandomList(d,1,572,5000)
l_train100 = CreateRandomList(d,1,572,100000)
l_test5 = CreateRandomList(d,573,715,5000)
l_test10 = CreateRandomList(d,573,715,10000)

torch.save('./TrainingList_100k',l_train100)
torch.save('./TrainingList_5k',l_train5)
torch.save('./TestList_5k',l_test5)
torch.save('./TestList_10k',l_test10)
--c = torch.Tensor(9):zero()
--for i=1,l_train5:size(1) do
--        c[l_train5[i][4]+2] = c[l_train5[i][4]+2]+1
--end
--
--print(c:mul(1/l_train5:size(1)))
