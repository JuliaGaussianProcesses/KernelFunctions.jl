"""
    FBMKernel(; h::Real=0.5)

Fractional Brownian motion kernel with Hurst index `h`.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the fractional Brownian motion kernel with
[Hurst index](https://en.wikipedia.org/wiki/Hurst_exponent#Generalized_exponent)
``h \\in [0,1]`` is defined as
```math
k(x, x'; h) =  \\frac{\\|x\\|_2^{2h} + \\|x'\\|_2^{2h} - \\|x - x'\\|^{2h}}{2}.
```
"""
struct FBMKernel{T<:Real} <: Kernel
    h::T

    function FBMKernel(h::Real)
        @check_args(FBMKernel, h, zero(h) ≤ h ≤ one(h), "h ∈ [0, 1]")
        return new{typeof(h)}(h)
    end
end

FBMKernel(; h::Real=0.5) = FBMKernel(h)

function ParameterHandling.flatten(::Type{T}, k::FBMKernel{S}) where {T<:Real,S<:Real}
    function unflatten_to_fbmkernel(v::Vector{T})
        h = S(logistic(only(v)))
        return FBMKernel(h)
    end
    return T[logit(k.h)], unflatten_to_fbmkernel
end

function (κ::FBMKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    modX = sum(abs2, x)
    modY = sum(abs2, y)
    modXY = sqeuclidean(x, y)
    h = only(κ.h)
    return (modX^h + modY^h - modXY^h) / 2
end

function (κ::FBMKernel)(x::Real, y::Real)
    return (abs2(x)^only(κ.h) + abs2(y)^only(κ.h) - abs2(x - y)^only(κ.h)) / 2
end

function Base.show(io::IO, κ::FBMKernel)
    return print(io, "Fractional Brownian Motion Kernel (h = ", only(κ.h), ")")
end

_fbm(modX, modY, modXY, h) = (modX^h + modY^h - modXY^h) / 2

_mod(x::AbstractVector{<:Real}) = abs2.(x)
_mod(x::AbstractVector{<:AbstractVector{<:Real}}) = sum.(abs2, x)
# two lines above could be combined into the second (dispatching on general AbstractVectors), but this (somewhat) more performant
_mod(x::ColVecs) = vec(sum(abs2, x.X; dims=1))
_mod(x::RowVecs) = vec(sum(abs2, x.X; dims=2))

function kernelmatrix(κ::FBMKernel, x::AbstractVector)
    modx = _mod(x)
    modx_wide = modx * ones(eltype(modx), 1, length(modx)) # ad perf hack -- is unit tested
    modxx = pairwise(SqEuclidean(), x)
    return _fbm.(modx_wide, modx_wide', modxx, only(κ.h))
end

function kernelmatrix!(K::AbstractMatrix, κ::FBMKernel, x::AbstractVector)
    modx = _mod(x)
    pairwise!(SqEuclidean(), K, x)
    K .= _fbm.(modx, modx', K, κ.h)
    return K
end

function kernelmatrix(κ::FBMKernel, x::AbstractVector, y::AbstractVector)
    modxy = pairwise(SqEuclidean(), x, y)
    modx_wide = _mod(x) * ones(eltype(modxy), 1, length(y)) # ad perf hack -- is unit tested
    mody_wide = _mod(y) * ones(eltype(modxy), 1, length(x)) # ad perf hack -- is unit tested
    return _fbm.(modx_wide, mody_wide', modxy, only(κ.h))
end

function kernelmatrix!(
    K::AbstractMatrix, κ::FBMKernel, x::AbstractVector, y::AbstractVector
)
    pairwise!(SqEuclidean(), K, x, y)
    K .= _fbm.(_mod(x), _mod(y)', K, κ.h)
    return K
end

function kernelmatrix_diag(κ::FBMKernel, x::AbstractVector)
    modx = _mod(x)
    modxx = colwise(SqEuclidean(), x)
    return _fbm.(modx, modx, modxx, only(κ.h))
end

function kernelmatrix_diag(κ::FBMKernel, x::AbstractVector, y::AbstractVector)
    modxy = colwise(SqEuclidean(), x, y)
    return _fbm.(_mod(x), _mod(y), modxy, only(κ.h))
end
