"""
RationalQuadraticKernel([ρ=1.0[,α=2.0]])
The rational-quadratic kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y)=(1+||x−y||²/α)^(-α)
```
where `α` is a shape parameter of the Euclidean distance. Check [`GammaRationalQuadraticKernel`](@ref) for a generalization.
"""
struct RationalQuadraticKernel{Tα<:Real} <: Kernel
    α::Tα
end

function RationalQuadraticKernel(α::Tα=2.0) where {Tα}
    @check_args(RationalQuadraticKernel, α, α > zero(Tα), "α > 1")
    return RationalQuadraticKernel{Tα}(α)
end

params(k::RationalQuadraticKernel) = (k.α)
opt_params(k::RationalQuadraticKernel) = (k.α)

@inline kappa(κ::RationalQuadraticKernel, d²::T) where {T<:Real} = (one(T)+d²/κ.α)^(-κ.α)

metric(::RationalQuadraticKernel) = SqEuclidean()

"""
`GammaRationalQuadraticKernel([ρ=1.0[,α=2.0[,γ=2.0]]])`
The Gamma-rational-quadratic kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y)=(1+ρ^(2γ)||x−y||^(2γ)/α)^(-α)
```
where `α` is a shape parameter of the Euclidean distance and `γ` is another shape parameter.
"""
struct GammaRationalQuadraticKernel{Tα<:Real, Tγ<:Real} <: Kernel
    α::Tα
    γ::Tγ
end

function GammaRationalQuadraticKernel(α::Tα=2.0, γ::Tγ=2*one(Tα)) where {Tα<:Real, Tγ<:Real}
    @check_args(GammaRationalQuadraticKernel, α, α > one(Tα), "α > 1")
    @check_args(GammaRationalQuadraticKernel, γ, γ >= one(Tγ), "γ >= 1")
    return GammaRationalQuadraticKernel{Tα, Tγ}(α, γ)
end

params(k::GammaRationalQuadraticKernel) = (k.α,k.γ)
opt_params(k::GammaRationalQuadraticKernel) = (k.α,k.γ)

@inline kappa(κ::GammaRationalQuadraticKernel, d²::T) where {T<:Real} = (one(T)+d²^κ.γ/κ.α)^(-κ.α)

metric(::GammaRationalQuadraticKernel) = SqEuclidean()
