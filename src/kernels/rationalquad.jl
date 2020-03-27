"""
    RationalQuadraticKernel(; α = 2.0)

The rational-quadratic kernel is a Mercer kernel given by the formula:
```
    κ(x,y)=(1+||x−y||²/α)^(-α)
```
where `α` is a shape parameter of the Euclidean distance. Check [`GammaRationalQuadraticKernel`](@ref) for a generalization.
"""
struct RationalQuadraticKernel{Tα<:Real} <: BaseKernel
    α::Vector{Tα}
    function RationalQuadraticKernel(;alpha::T=2.0, α::T=alpha) where {T}
        @check_args(RationalQuadraticKernel, α, α > zero(T), "α > 1")
        return new{T}([α])
    end
end

kappa(κ::RationalQuadraticKernel, d²::T) where {T<:Real} = (one(T)+d²/first(κ.α))^(-first(κ.α))
metric(::RationalQuadraticKernel) = SqEuclidean()

Base.show(io::IO, κ::RationalQuadraticKernel) = print(io, "Rational Quadratic Kernel (α = $(first(κ.α)))")

"""
`GammaRationalQuadraticKernel([ρ=1.0[,α=2.0[,γ=2.0]]])`
The Gamma-rational-quadratic kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y)=(1+ρ^(2γ)||x−y||^(2γ)/α)^(-α)
```
where `α` is a shape parameter of the Euclidean distance and `γ` is another shape parameter.
"""
struct GammaRationalQuadraticKernel{Tα<:Real, Tγ<:Real} <: BaseKernel
    α::Vector{Tα}
    γ::Vector{Tγ}
    function GammaRationalQuadraticKernel(;alpha::Tα=2.0, gamma::Tγ=2.0, α::Tα=alpha, γ::Tγ=gamma) where {Tα<:Real, Tγ<:Real}
        @check_args(GammaRationalQuadraticKernel, α, α > one(Tα), "α > 1")
        @check_args(GammaRationalQuadraticKernel, γ, γ >= one(Tγ), "γ >= 1")
        return new{Tα, Tγ}([α], [γ])
    end
end

kappa(κ::GammaRationalQuadraticKernel, d²::T) where {T<:Real} = (one(T)+d²^first(κ.γ)/first(κ.α))^(-first(κ.α))
metric(::GammaRationalQuadraticKernel) = SqEuclidean()

Base.show(io::IO, κ::RationalQuadraticKernel) = print(io, "Gamma Rational Quadratic Kernel (α = $(first(κ.α)), γ = $(first(κ.γ)))")
