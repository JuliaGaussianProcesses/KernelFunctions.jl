"""
    FBMKernel(; h::Real=0.5)

Fractional Brownian motion kernel with Hurst index h from (0,1) given by
```
    κ(x,y) =  ( |x|²ʰ + |y|²ʰ - |x-y|²ʰ ) / 2
```

For h=1/2, this is the Wiener Kernel, for h>1/2, the increments are
positively correlated and for h<1/2 the increments are negatively correlated.
"""
struct FBMKernel{T<:Real} <: BaseKernel
    h::T
    function FBMKernel(;h::T=0.5) where {T<:Real}
        @assert h<=1.0 && h>=0.0 "FBMKernel: Given Hurst index h is invalid."
        return new{T}(h)
    end
end

_fbm(modX, modY, modXY, h) = (modX^h + modY^h - modXY^h)/2

function kernelmatrix(κ::FBMKernel, X::AbstractMatrix; obsdim::Int = defaultobs)
    @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    modX = sum(abs2, X; dims = 3 - obsdim)
    modXX = pairwise(SqEuclidean(), X, dims = obsdim)
    return _fbm.(vec(modX), reshape(modX, 1, :), modXX, κ.h)
end

function kernelmatrix!(K::AbstractMatrix, κ::FBMKernel, X::AbstractMatrix; obsdim::Int = defaultobs)
    K = kernelmatrix(κ, X; obsdim = obsdim)
end

function kernelmatrix(
    κ::FBMKernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    
    modX = sum(abs2, X, dims=3-obsdim)
    modY = sum(abs2, Y, dims=3-obsdim)
    modXY = pairwise(SqEuclidean(), X, Y,dims=obsdim)
    return _fbm.(vec(modX), reshape(modY, 1, :), modXY, κ.h)
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::FBMKernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    K = kernelmatrix(κ, X, Y; obsdim = obsdim)
end

#Syntactic Sugar
function (κ::FBMKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    modX = sum(abs2, x)
    modY = sum(abs2, y)
    modXY = sqeuclidean(x, y)
    return (modX^κ.h + modY^κ.h - modXY^κ.h)/2
end

(κ::FBMKernel)(x::Real, y::Real) = (abs2(x)^κ.h + abs2(y)^κ.h - abs2(x-y)^κ.h)/2

function (κ::FBMKernel)(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; obsdim::Integer=defaultobs)
    return kernelmatrix(κ, X, Y, obsdim=obsdim)
end

function (κ::FBMKernel)(X::AbstractMatrix{<:Real}; obsdim::Integer=defaultobs)
    return kernelmatrix(κ, X, obsdim=obsdim)
end
