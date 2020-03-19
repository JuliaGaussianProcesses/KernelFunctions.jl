"""
    FBMKernel(; h::Real=0.5)
Fractional Brownian motion kernel with Hurst index h from (0,1) given by
```
    κ(x,y) =  ( |x|²ʰ + |y|²ʰ - |x-y|²ʰ ) / 2
```

For h=1/2, this is the Wiener Kernel, for h>1/2, the increments are
positively correlated and for h<1/2 the increments are negatively correlated.
%
"""
struct FBMKernel{T<:Real} <: BaseKernel
    h::T
    function FBMKernel(;h::T=0.5) where {T<:Real}
        @assert h<=1.0 && h>=0.0 "FBMKernel: Given Hurst index h is invalid."
        new{T}(h)
    end
end

kappa(κ::FBMKernel, d::Real) = error("Not Implemented: Please use `kernelmatrix` or `kerneldiagmatrix` instead.")

_fbm(modX, modY, modXY, h) = (modX^h + modY^h - modXY^h)/2

function kernelmatrix(κ::FBMKernel, X::AbstractMatrix; obsdim::Int = defaultobs)
    @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    K = map(modX->_fbm(modX, modX, 0, κ.h), pairwise(SqEuclidean(),X,dims=obsdim))
end

function kernelmatrix(
    κ::FBMKernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    
    modX = pairwise(SqEuclidean(),X,dims=obsdim)
    modY = pairwise(SqEuclidean(),Y,dims=obsdim)
    modXY = pairwise(SqEuclidean(),X-Y,dims=obsdim)
    K = map((x)->_fbm(x[1], x[2], x[3], κ.h), zip(modX, modY, modXY))
end

#Syntactic Sugar
function (κ::FBMKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    modX = evaluate(SqEuclidean(), x, zero(x))
    modY = evaluate(SqEuclidean(), y, zero(y))
    modXY = evaluate(SqEuclidean(), x-y, zero(x-y))
    (modX^κ.h + modY^κ.h - modXY^κ.h)/2
end

(κ::FBMKernel)(X::AbstractMatrix{T}, Y::AbstractMatrix{T}; obsdim::Integer=defaultobs) where {T} = kernelmatrix(κ, X, Y, obsdim=obsdim)
(κ::FBMKernel)(X::AbstractMatrix{T}; obsdim::Integer=defaultobs) where {T} = kernelmatrix(κ, X, obsdim=obsdim)
