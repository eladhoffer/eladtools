require 'image'
lmdb = require 'lmdb'

env, msg = lmdb.environment('testDB',{subdir = false, max_dbs = 8})
db_num = env:db_open('testImage')

function ReadImage(num)
    local data
    env:transaction(function(txn)
                    data = txn:get(num)
                end
                ,lmdb.READ_ONLY, db_num )
    return torch.deserialize(tostring(data))
end



function WriteImage(num,data)
    local serialized = torch.serialize(data)
    env:transaction(function(txn)
        txn:put(num,serialized)
    end
    ,lmdb.WRITE, db_num )
end




