"""
RationalQuadraticKernel([ρ=1.0[,α=2.0]])
The rational-quadratic kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y)=(1+ρ²||x−y||²/α)^(-α)
```
where `α` is a shape parameter of the Euclidean distance. Check [`GammaRationalQuadraticKernel`](@ref) for a generalization.
"""
struct RationalQuadraticKernel{Tr,Tα<:Real} <: Kernel{Tr}
    transform::Tr
    α::Tα
    function RationalQuadraticKernel{Tr, Tα}(t::Tr, α::Tα) where {Tr, Tα}
        @check_args(RationalQuadraticKernel, α, α > zero(Tα), "α > 1")
        return new{Tr, Tα}(t, α)
    end
end

function RationalQuadraticKernel(ρ::Real=1.0, α::Real=2.0)
    RationalQuadraticKernel(ScaleTransform(ρ),α)
end

function RationalQuadraticKernel(ρ::AbstractVector{<:Real}, α::Real=2.0)
    RationalQuadraticKernel(ARDTransform(ρ), α)
end

function RationalQuadraticKernel(t::Tr, α::Tα=2.0) where {Tr<:Transform, Tα<:Real}
    return RationalQuadraticKernel{Tr, Tα}(t, α)
end


params(k::RationalQuadraticKernel) = (params(transform(k)),k.α)
opt_params(k::RationalQuadraticKernel) = (opt_params(transform(k)),k.α)

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
struct GammaRationalQuadraticKernel{Tr, Tα<:Real, Tγ<:Real} <: Kernel{Tr}
    transform::Tr
    α::Tα
    γ::Tγ
    function GammaRationalQuadraticKernel{Tr,Tα,Tγ}(t::Tr, α::Tα, γ::Tγ) where {Tr, Tα<:Real, Tγ<:Real}
        @check_args(GammaRationalQuadraticKernel, α, α > one(Tα), "α > 1")
        @check_args(GammaRationalQuadraticKernel, γ, γ >= one(Tγ), "γ >= 1")
        return new{Tr, Tα, Tγ}(t, α, γ)
    end
end

function GammaRationalQuadraticKernel(ρ::Real=1.0, α::Real=2.0, γ::Real=2.0)
    GammaRationalQuadraticKernel(ScaleTransform(ρ), α, γ)
end

function GammaRationalQuadraticKernel(ρ::AbstractVector{<:Real}, α::Real=2.0, γ::Real=2.0)
    GammaRationalQuadraticKernel(ARDTransform(ρ),α,γ)
end

function GammaRationalQuadraticKernel(t::Tr,α::Tα=2.0,γ::Tγ=2.0) where {Tr<:Transform, Tα<:Real, Tγ<:Real}
    GammaRationalQuadraticKernel{Tr, Tα, Tγ}(t, α, γ)
end

params(k::GammaRationalQuadraticKernel) = (params(k.transform),k.α,k.γ)
opt_params(k::GammaRationalQuadraticKernel) = (opt_params(k.transform),k.α,k.γ)

@inline kappa(κ::GammaRationalQuadraticKernel, d²::T) where {T<:Real} = (one(T)+d²^κ.γ/κ.α)^(-κ.α)

metric(::GammaRationalQuadraticKernel) = SqEuclidean()
