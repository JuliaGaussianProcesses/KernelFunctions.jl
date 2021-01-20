"""
    RationalKernel(; α::Real=2.0)

Rational kernel with shape parameter `α`.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the rational kernel with shape parameter
``\\alpha > 0`` is defined as
```math
k(x, x'; \\alpha) = \\bigg(1 + \\frac{\\|x - x'\\|_2}{\\alpha}\\bigg)^{-\\alpha}.
```

The [`ExponentialKernel`](@ref) is recovered in the limit as ``\\alpha \\to \\infty``.

See also: [`GammaRationalKernel`](@ref)
"""
struct RationalKernel{Tα<:Real} <: SimpleKernel
    α::Vector{Tα}
    function RationalKernel(; alpha::T=2.0, α::T=alpha) where {T}
        @check_args(RationalKernel, α, α > zero(T), "α > 0")
        return new{T}([α])
    end
end

@functor RationalKernel

function kappa(κ::RationalKernel, d::Real)
    return (one(d) + d / first(κ.α))^(-first(κ.α))
end

metric(::RationalKernel) = Euclidean()

function Base.show(io::IO, κ::RationalKernel)
    return print(io, "Rational Kernel (α = $(first(κ.α)))")
end

"""
    RationalQuadraticKernel(; α::Real=2.0)

Rational-quadratic kernel with shape parameter `α`.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the rational-quadratic kernel with shape parameter
``\\alpha > 0`` is defined as
```math
k(x, x'; \\alpha) = \\bigg(1 + \\frac{\\|x - x'\\|_2^2}{2\\alpha}\\bigg)^{-\\alpha}.
```

The [`SqExponentialKernel`](@ref) is recovered in the limit as ``\\alpha \\to \\infty``.

See also: [`GammaRationalKernel`](@ref)
"""
struct RationalQuadraticKernel{Tα<:Real} <: SimpleKernel
    α::Vector{Tα}
    function RationalQuadraticKernel(; alpha::T=2.0, α::T=alpha) where {T}
        @check_args(RationalQuadraticKernel, α, α > zero(T), "α > 0")
        return new{T}([α])
    end
end

@functor RationalQuadraticKernel

function kappa(κ::RationalQuadraticKernel, d²::T) where {T<:Real}
    return (one(T) + d² / (2 * first(κ.α)))^(-first(κ.α))
end

metric(::RationalQuadraticKernel) = SqEuclidean()

function Base.show(io::IO, κ::RationalQuadraticKernel)
    return print(io, "Rational Quadratic Kernel (α = $(first(κ.α)))")
end

"""
    GammaRationalKernel(; α::Real=2.0, γ::Real=2.0)

γ-rational kernel with shape parameters `α` and `γ`.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the γ-rational kernel with shape
parameters ``\\alpha > 0`` and ``\\gamma \\in (0, 2]`` is defined as
```math
k(x, x'; \\alpha, \\gamma) = \\bigg(1 + \\frac{\\|x - x'\\|_2^{\\gamma}}{\\alpha}\\bigg)^{-\\alpha}.
```

The [`GammaExponentialKernel`](@ref) is recovered in the limit as ``\\alpha \\to \\infty``.

See also: [`RationalKernel`](@ref), [`RationalQuadraticKernel`](@ref)
"""
struct GammaRationalKernel{Tα<:Real,Tγ<:Real} <: SimpleKernel
    α::Vector{Tα}
    γ::Vector{Tγ}
    function GammaRationalKernel(;
        alpha::Real=2.0, gamma::Real=2.0, α::Real=alpha, γ::Real=gamma
    )
        @check_args(GammaRationalKernel, α, α > zero(α), "α > 0")
        @check_args(GammaRationalKernel, γ, zero(γ) < γ ≤ 2, "γ ∈ (0, 2]")
        return new{typeof(α),typeof(γ)}([α], [γ])
    end
end

@functor GammaRationalKernel

function kappa(κ::GammaRationalKernel, d::Real)
    return (one(d) + d^first(κ.γ) / first(κ.α))^(-first(κ.α))
end

metric(::GammaRationalKernel) = Euclidean()

function Base.show(io::IO, κ::GammaRationalKernel)
    return print(io, "Gamma Rational Kernel (α = $(first(κ.α)), γ = $(first(κ.γ)))")
end
