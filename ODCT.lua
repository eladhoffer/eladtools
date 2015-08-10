local function kron(A,B)
    local m, n = A:size(1), A:size(2)
    local p, q = B:size(1), B:size(2)
    local C = torch.Tensor(m*p,n*q)

    for i=1,m do
        for j=1,n do
            C[{{(i-1)*p+1,i*p},{(j-1)*q+1,j*q}}] = torch.mul(B, A[i][j])
        end
    end
    return C
end


local function odctdict(n,L)
    D = torch.zeros(n,L)
    D[{{},1}]:fill(1/math.sqrt(n))
    for k=2,L do
        local v = torch.linspace(0,n-1,n):mul((k-1)*math.pi/L):cos()
        v:add(-v:mean())
        v:div(v:norm())
        D[{{},k}]:copy(v)
    end
    return D
end

local function odct2dict(w,h,L)

    n = math.ceil(math.sqrt(L))
    D = odctdict(w,n)
    D = kron(D,odctdict(h,n))
    return D
end

local function odct3dict(ndim,w,h,L)
    local l = math.ceil(math.pow(L,1/3))
    local sz = ndim*w*h
    local n = torch.FloatTensor({ndim,w,h}):pow(3):div(sz):pow(1/2):mul(L):pow(1/3):ceil()



    D = odctdict(ndim,n[1]);

    D = kron(D,odctdict(w,n[2]))
    D = kron(D,odctdict(h,n[3]))
    return D--:narrow(2,1,L):contiguous()
end
