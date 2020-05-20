"""
    FBMKernel(; h::Real=0.5)

Fractional Brownian motion kernel with Hurst index `h` from (0,1) given by
```
    κ(x,y) =  ( |x|²ʰ + |y|²ʰ - |x-y|²ʰ ) / 2
```

For `h=1/2`, this is the Wiener Kernel, for `h>1/2`, the increments are
positively correlated and for `h<1/2` the increments are negatively correlated.
"""
struct FBMKernel{T<:Real} <: BaseKernel
    h::Vector{T}
    function FBMKernel(; h::T=0.5) where {T<:Real}
        @assert 0.0 <= h <= 1.0 "FBMKernel: Given Hurst index h is invalid."
        return new{T}([h])
    end
end

function (κ::FBMKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    modX = sum(abs2, x)
    modY = sum(abs2, y)
    modXY = sqeuclidean(x, y)
    h = first(κ.h)
    return (modX^h + modY^h - modXY^h)/2
end

(κ::FBMKernel)(x::Real, y::Real) = (abs2(x)^first(κ.h) + abs2(y)^first(κ.h) - abs2(x - y)^first(κ.h)) / 2

Base.show(io::IO, κ::FBMKernel) = print(io, "Fractional Brownian Motion Kernel (h = ", first(κ.h), ")")

_fbm(modX, modY, modXY, h) = (modX^h + modY^h - modXY^h)/2

_mod(x::AbstractVector{<:Real}) = abs2.(x)
_mod(x::ColVecs) = vec(sum(abs2, x.X; dims=1))
_mod(x::RowVecs) = vec(sum(abs2, x.X; dims=2))

function kernelmatrix(κ::FBMKernel, x::AbstractVector)
    modx = _mod(x)
    modxx = pairwise(SqEuclidean(), x)
    return _fbm.(modx, modx', modxx, κ.h)
end

function kernelmatrix!(K::AbstractMatrix, κ::FBMKernel, x::AbstractVector)
    modx = _mod(x)
    pairwise!(K, SqEuclidean(), x)
    K .= _fbm.(modx, modx', K, κ.h)
    return K
end

function kernelmatrix(κ::FBMKernel, x::AbstractVector, y::AbstractVector)
    modxy = pairwise(SqEuclidean(), x, y)
    return _fbm.(_mod(x), _mod(y)', modxy, κ.h)
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::FBMKernel,
    x::AbstractVector,
    y::AbstractVector,
)
    pairwise!(K, SqEuclidean(), x, y)
    K .= _fbm.(_mod(x), _mod(y)', K, κ.h)
    return K
end
