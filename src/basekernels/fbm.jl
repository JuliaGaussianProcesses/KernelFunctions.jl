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
    modXY = evaluate(SqEuclidean(sqroundoff), x, y)
    h = first(κ.h)
    return (modX^h + modY^h - modXY^h)/2
end

(κ::FBMKernel)(x::Real, y::Real) = (abs2(x)^first(κ.h) + abs2(y)^first(κ.h) - abs2(x - y)^first(κ.h)) / 2

Base.show(io::IO, κ::FBMKernel) = print(io, "Fractional Brownian Motion Kernel (h = ", first(κ.h), ")")

const sqroundoff = 1e-15

_fbm(modX, modY, modXY, h) = (modX^h + modY^h - modXY^h)/2

_mod(x::AbstractVector{<:Real}) = abs2.(x)
_mod(x::ColVecs) = vec(sum(abs2, x.X; dims=1))
_mod(x::RowVecs) = vec(sum(abs2, x.X; dims=2))

function kernelmatrix(κ::FBMKernel, x::AbstractVector)
    modx = _mod(x)
    modxx = pairwise(SqEuclidean(sqroundoff), x)
    return _fbm.(modx, modx', modxx, κ.h)
end

function kernelmatrix!(K::AbstractMatrix, κ::FBMKernel, x::AbstractVector)
    modx = _mod(x)
    modxx = pairwise(SqEuclidean(sqroundoff), x)
    K .= _fbm.(modx, reshape(modx, 1, :), modxx, κ.h)
    return K
end

function kernelmatrix(κ::FBMKernel, x::AbstractVector, y::AbstractVector)
    modxy = pairwise(SqEuclidean(sqroundoff), x, y)
    return _fbm.(_mod(x), reshape(_mod(y), 1, :), modxy, κ.h)
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::FBMKernel,
    X::AbstractVector,
    Y::AbstractVector,
)
    modxy = pairwise(SqEuclidean(sqroundoff), X, Y,dims = obsdim)
    K .= _fbm.(_mod(x), reshape(_mod(y), 1, :), modxy, κ.h)
    return K
end
