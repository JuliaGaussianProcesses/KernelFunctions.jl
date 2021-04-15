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
    h::Vector{T}
    function FBMKernel(; h::Real=0.5)
        @check_args(FBMKernel, h, zero(h) ≤ h ≤ one(h), "h ∈ [0, 1]")
        return new{typeof(h)}([h])
    end
end

@functor FBMKernel

function (κ::FBMKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    modX = sum(abs2, x)
    modY = sum(abs2, y)
    modXY = sqeuclidean(x, y)
    h = first(κ.h)
    return (modX^h + modY^h - modXY^h) / 2
end

function (κ::FBMKernel)(x::Real, y::Real)
    return (abs2(x)^first(κ.h) + abs2(y)^first(κ.h) - abs2(x - y)^first(κ.h)) / 2
end

function Base.show(io::IO, κ::FBMKernel)
    return print(io, "Fractional Brownian Motion Kernel (h = ", first(κ.h), ")")
end

_fbm(modX, modY, modXY, h) = (modX^h + modY^h - modXY^h) / 2

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
    K::AbstractMatrix, κ::FBMKernel, x::AbstractVector, y::AbstractVector
)
    pairwise!(K, SqEuclidean(), x, y)
    K .= _fbm.(_mod(x), _mod(y)', K, κ.h)
    return K
end

function kernelmatrix_diag(κ::FBMKernel, x::AbstractVector)
    modx = _mod(x)
    modxx = colwise(SqEuclidean(), x)
    return _fbm.(modx, modx, modxx, κ.h)
end

function kernelmatrix_diag(κ::FBMKernel, x::AbstractVector, y::AbstractVector)
    modxy = colwise(SqEuclidean(), x, y)
    return _fbm.(_mod(x), _mod(y), modxy, κ.h)
end
