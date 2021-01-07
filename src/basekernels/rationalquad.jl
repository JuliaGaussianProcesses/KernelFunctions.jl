"""
    RationalQuadraticKernel(; α=2.0)

The rational-quadratic kernel is a Mercer kernel given by the formula:
```
    κ(x, y) = (1 + ||x − y||² / (2α))^(-α)
```
where `α` is a shape parameter of the Euclidean distance. Check
[`GammaRationalQuadraticKernel`](@ref) for a generalization.
"""
struct RationalQuadraticKernel{Tα<:Real} <: SimpleKernel
    α::Vector{Tα}
    function RationalQuadraticKernel(; alpha::T = 2.0, α::T = alpha) where {T}
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
    print(io, "Rational Quadratic Kernel (α = $(first(κ.α)))")
end

"""
`GammaRationalQuadraticKernel([α=2.0 [, γ=2.0]])`

The Gamma-rational-quadratic kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x, y) = (1 + ||x−y||^γ / α)^(-α)
```
where `α` is a shape parameter of the Euclidean distance and `γ` is another shape parameter.
"""
struct GammaRationalQuadraticKernel{Tα<:Real,Tγ<:Real} <: SimpleKernel
    α::Vector{Tα}
    γ::Vector{Tγ}
    function GammaRationalQuadraticKernel(;
        alpha::Tα = 2.0,
        gamma::Tγ = 2.0,
        α::Tα = alpha,
        γ::Tγ = gamma,
    ) where {Tα<:Real,Tγ<:Real}
        @check_args(GammaRationalQuadraticKernel, α, α > zero(Tα), "α > 0")
        @check_args(GammaRationalQuadraticKernel, γ, zero(γ) < γ <= 2, "0 < γ <= 2")
        return new{Tα,Tγ}([α], [γ])
    end
end

@functor GammaRationalQuadraticKernel

function kappa(κ::GammaRationalQuadraticKernel, d²::Real)
    return (one(d²) + d²^(first(κ.γ) / 2) / first(κ.α))^(-first(κ.α))
end

metric(::GammaRationalQuadraticKernel) = SqEuclidean()

function Base.show(io::IO, κ::GammaRationalQuadraticKernel)
    print(io, "Gamma Rational Quadratic Kernel (α = $(first(κ.α)), γ = $(first(κ.γ)))")
end
