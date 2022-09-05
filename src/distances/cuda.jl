
const CuSubArray{T} = SubArray{T, <:Any, <:CuArray}
const CuColVecs{T} = ColVecs{T, <:Union{CuMatrix{T}, CuSubArray{T}}}
const CuRowVecs{T} = RowVecs{T, <:Union{CuMatrix{T}, CuSubArray{T}}}

const _CuVector{T} = Union{CuVector{T}, SubArray{T, 1, <:CuArray}}

# SqEuclidean

# Vector{<:Real}

pairwise(::SqEuclidean, x::_CuVector{<:Real}) = (x .- x').^2

pairwise(::SqEuclidean, x::_CuVector{T}, y::_CuVector{T}) where {T<:Real} = (x .- y').^2

colwise(::SqEuclidean, x::_CuVector{T}) where {T<:Real} = CUDA.zeros(T, length(x))

colwise(::SqEuclidean, x::_CuVector{T}, y::_CuVector{T}) where {T<:Real} = (x .- y).^2

# ColVecs

function pairwise(::SqEuclidean, x::CuColVecs{<:Real})
    X = CuArray(x.X) # needed to handle subarrays
    X_ = sum(abs2, X; dims=1)
    return X_ .+ X_' .- 2 .* (X' * X)
end

function pairwise(::SqEuclidean, x::CuColVecs{T}, y::CuColVecs{T}) where {T<:Real}
    X = CuArray(x.X) # needed to handle subarrays
    Y = CuArray(y.X) # needed to handle subarrays
    X_ = sum(abs2, X; dims=1)
    Y_ = sum(abs2, Y; dims=1)
    return X_' .+ Y_ .- 2 .* (X' * Y)
end

function colwise(::SqEuclidean, x::CuColVecs{T}) where {T<:Real}
    return CUDA.zeros(T, length(x))
end

function colwise(::SqEuclidean, x::CuColVecs{T}, y::CuColVecs{T}) where {T<:Real}
    return vec(sum(abs2, x.X - y.X; dims=1))
end

# RowVecs

function pairwise(::SqEuclidean, x::CuRowVecs{<:Real})
    X = CuArray(x.X)
    X_ = sum(abs2, X; dims=2)
    return X_' .+ X_ .- 2 .* (X * X')
end

function pairwise(::SqEuclidean, x::CuRowVecs{T}, y::CuRowVecs{T}) where {T<:Real}
    X = CuArray(x.X) # needed to handle subarrays
    Y = CuArray(y.X) # needed to handle subarrays
    X_ = sum(abs2, X; dims=2)
    Y_ = sum(abs2, Y; dims=2)
    return X_ .+ Y_' .- 2 .* (X * Y')
end

colwise(::SqEuclidean, x::CuRowVecs{T}) where {T<:Real} = CUDA.zeros(T, length(x))

function colwise(::SqEuclidean, x::CuRowVecs{T}, y::CuRowVecs{T}) where {T<:Real}
    return vec(sum(abs2, x.X - y.X; dims=2))
end

# Eucldiean: Derive from SqEuclidean implementations.

const GPUVecType{T} = Union{_CuVector{T}, CuColVecs{T}, CuRowVecs{T}}

pairwise(::Euclidean, x::GPUVecType{<:Real}) = sqrt.(pairwise(SqEuclidean(), x))

function pairwise(::Euclidean, x::GPUVecType{T}, y::GPUVecType{T}) where {T<:Real}
    return sqrt.(pairwise(SqEuclidean(), x, y))
end

colwise(::Euclidean, x::GPUVecType{T}) where {T<:Real} = CUDA.zeros(T, length(x))

function colwise(::Euclidean, x::GPUVecType{T}, y::GPUVecType{T}) where {T<:Real}
    return sqrt.(colwise(SqEuclidean(), x, y))
end

#
# DotProduct
#

# Vector
pairwise(::DotProduct, x::_CuVector{<:Real}) = x * x'

pairwise(::DotProduct, x::_CuVector{T}, y::_CuVector{T}) where {T<:Real} = x * y'

colwise(::DotProduct, x::_CuVector{<:Real}) = x.^2

colwise(::DotProduct, x::_CuVector{T}, y::_CuVector{T}) where {T<:Real} = x .* y

# ColVecs
pairwise(::DotProduct, x::CuColVecs{<:Real}) = x.X' * x.X

function pairwise(::DotProduct, x::CuColVecs{T}, y::CuColVecs{T}) where {T<:Real}
    return x.X' * y.X
end

function colwise(::DotProduct, x::CuColVecs{<:Real})
    return vec(sum(x.X.^2; dims=1))
end

function colwise(::DotProduct, x::CuColVecs{T}, y::CuColVecs{T}) where {T<:Real}
    return vec(sum(x.X .* y.X; dims=1))
end

# RowVecs
pairwise(::DotProduct, x::CuRowVecs{<:Real}) = x.X * x.X'

function pairwise(::DotProduct, x::CuRowVecs{T}, y::CuRowVecs{T}) where {T<:Real}
    return x.X * y.X'
end

function colwise(::DotProduct, x::CuRowVecs{<:Real})
    return vec(sum(x.X.^2; dims=2))
end

function colwise(::DotProduct, x::CuRowVecs{T}, y::CuRowVecs{T}) where {T<:Real}
    return vec(sum(x.X .* y.X; dims=2))
end
