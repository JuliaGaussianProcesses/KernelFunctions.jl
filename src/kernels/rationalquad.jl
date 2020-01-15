"""
RationalQuadraticKernel([ρ=1.0[,α=2.0]])
The rational-quadratic kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y)=(1+ρ²||x−y||²/α)^(-α)
```
where `α` is a shape parameter of the Euclidean distance. Check [`GammaRationalQuadraticKernel`](@ref) for a generalization.
"""
struct RationalQuadraticKernel{T,Tr,Tα<:Real} <: Kernel{T,Tr}
    transform::Tr
    α::Tα

    function RationalQuadraticKernel{T,Tr,Tα}(t::Tr,α::Tα) where {T,Tr,Tα<:Real}
        new{T,Tr,Tα}(t,α)
    end
end

function RationalQuadraticKernel(ρ::T₁=1.0,α::T₂=2.0) where {T₁<:Real,T₂<:Real}
    @check_args(RationalQuadraticKernel, α, α > zero(T₂), "α > 1")
    RationalQuadraticKernel{T₁,ScaleTransform{T₁},T₂}(ScaleTransform(ρ),α)
end

function RationalQuadraticKernel(ρ::AbstractVector{T₁},α::T₂=2.0) where {T₁<:Real,T₂<:Real}
    @check_args(RationalQuadraticKernel, α, α > zero(T₂), "α > 1")
    RationalQuadraticKernel{T₁,ARDTransform{T₁,length(ρ)},T₂}(ARDTransform(ρ),α)
end

function RationalQuadraticKernel(t::Tr,α::T=2.0) where {Tr<:Transform,T<:Real}
    @check_args(RationalQuadraticKernel, α, α > zero(T), "α > 1")
    RationalQuadraticKernel{eltype(t),Tr,T}(t,α)
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
struct GammaRationalQuadraticKernel{T,Tr,Tα<:Real,Tγ<:Real} <: Kernel{T,Tr}
    transform::Tr
    α::Tα
    γ::Tγ

    function GammaRationalQuadraticKernel{T,Tr,Tα,Tγ}(t::Tr,α::Tα,γ::Tγ) where {T,Tr,Tα<:Real,Tγ<:Real}
        new{T,Tr,Tα,Tγ}(t,α,γ)
    end
end

function GammaRationalQuadraticKernel(ρ::T₁=1.0,α::T₂=2.0,γ::T₃=2.0) where {T₁<:Real,T₂<:Real,T₃<:Real}
    @check_args(GammaRationalQuadraticKernel, α, α > one(T₂), "α > 1")
    @check_args(GammaRationalQuadraticKernel, γ, γ >= one(T₂), "γ >= 1")
    GammaRationalQuadraticKernel{T₁,ScaleTransform{T₁},T₂,T₃}(ScaleTransform(ρ),α,γ)
end

function GammaRationalQuadraticKernel(ρ::AbstractVector{T₁},α::T₂=2.0,γ::T₃=2.0) where {T₁<:Real,T₂<:Real,T₃<:Real}
    @check_args(GammaRationalQuadraticKernel, α, α > one(T₂), "α > 1")
    @check_args(GammaRationalQuadraticKernel, γ, γ >= one(T₃), "γ >= 1")
    GammaRationalQuadraticKernel{T₁,ARDTransform{T₁,length(ρ)},T₂,T₃}(ARDTransform(ρ),α,γ)
end

function GammaRationalQuadraticKernel(t::Tr,α::T₁=2.0,γ::T₂=2.0) where {Tr<:Transform,T₁<:Real,T₂<:Real}
    @check_args(GammaRationalQuadraticKernel, α, α > one(T₁), "α > 1")
    @check_args(GammaRationalQuadraticKernel, γ, γ >= one(T₂), "γ >= 1")
    GammaRationalQuadraticKernel{eltype(t),Tr,T₁,T₂}(t,α,γ)
end

params(k::GammaRationalQuadraticKernel) = (params(k.transform),k.α,k.γ)
opt_params(k::GammaRationalQuadraticKernel) = (opt_params(k.transform),k.α,k.γ)

@inline kappa(κ::GammaRationalQuadraticKernel, d²::T) where {T<:Real} = (one(T)+d²^κ.γ/κ.α)^(-κ.α)

metric(::GammaRationalQuadraticKernel) = SqEuclidean()