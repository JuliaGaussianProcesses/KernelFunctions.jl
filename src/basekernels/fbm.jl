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

Base.show(io::IO, κ::FBMKernel) = print(io, "Fractional Brownian Motion Kernel (h = ", first(κ.h), ")")

const sqroundoff = 1e-15

_fbm(modX, modY, modXY, h) = (modX^h + modY^h - modXY^h)/2

function kernelmatrix(κ::FBMKernel, X::AbstractMatrix; obsdim::Int = defaultobs)
    @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    modX = sum(abs2, X; dims = feature_dim(obsdim))
    modXX = pairwise(SqEuclidean(sqroundoff), X, dims = obsdim)
    return _fbm.(vec(modX), reshape(modX, 1, :), modXX, κ.h)
end

function kernelmatrix!(K::AbstractMatrix, κ::FBMKernel, X::AbstractMatrix; obsdim::Int = defaultobs)
    @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    modX = sum(abs2, X; dims = feature_dim(obsdim))
    modXX = pairwise(SqEuclidean(sqroundoff), X, dims = obsdim)
    K .= _fbm.(vec(modX), reshape(modX, 1, :), modXX, κ.h)
    return K
end

function kernelmatrix(
    κ::FBMKernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    modX = sum(abs2, X, dims = feature_dim(obsdim))
    modY = sum(abs2, Y, dims = feature_dim(obsdim))
    modXY = pairwise(SqEuclidean(sqroundoff), X, Y,dims = obsdim)
    return _fbm.(vec(modX), reshape(modY, 1, :), modXY, κ.h)
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::FBMKernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    modX = sum(abs2, X, dims = feature_dim(obsdim))
    modY = sum(abs2, Y, dims = feature_dim(obsdim))
    modXY = pairwise(SqEuclidean(sqroundoff), X, Y,dims = obsdim)
    K .= _fbm.(vec(modX), reshape(modY, 1, :), modXY, κ.h)
    return K
end

function kappa(κ::FBMKernel, x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    modX = sum(abs2, x)
    modY = sum(abs2, y)
    modXY = evaluate(SqEuclidean(sqroundoff), x, y)
    h = first(κ.h)
    return (modX^h + modY^h - modXY^h)/2
end

(κ::FBMKernel)(x::Real, y::Real) = (abs2(x)^first(κ.h) + abs2(y)^first(κ.h) - abs2(x-y)^first(κ.h))/2
