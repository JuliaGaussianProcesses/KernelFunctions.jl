"""
    RationalQuadraticKernel([ρ=1.0[,α=2.0]])

    The rational-quadratic kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y)=(1+||x−y||^2/α)^(-α)
```
    where `α` is a shape parameter of the Euclidean distance. Check `GammaRationalQuadraticKernel` for a generalization.
"""
struct RationalQuadraticKernel{T,Tr,Tα<:Real} <: Kernel{T,Tr}
    transform::Tr
    metric::SqEuclidean
    α::Tα
    function RationalQuadraticKernel{T,Tr,Tα}(t::Tr,α::Tα) where {T,Tr,Tα<:Real}
        new{T,Tr,Tα}(t,SqEuclidean(),α)
    end
end

function RationalQuadraticKernel(ρ::T₁=1.0,α::T₂=2.0) where {T₁<:Real,T₂<:Real}
    @check_args(RationalQuadraticKernel, α, α > zero(T₂), "α > 1")
    RationalQuadraticKernel{T₁,ScaleTransform{T₁},T₂}(ScaleTransform(ρ),α)
end

function RationalQuadraticKernel(ρ::A,α::T=2.0) where {A<:AbstractVector{<:Real},T<:Real}
    @check_args(RationalQuadraticKernel, α, α > zero(T), "α > 1")
    RationalQuadraticKernel{eltype(A),ScaleTransform{A},T}(ScaleTransform(ρ),α)
end

function RationalQuadraticKernel(t::Tr,α::T=2.0) where {Tr<:Transform,T<:Real}
    @check_args(RationalQuadraticKernel, α, α > zero(T), "α > 1")
    RationalQuadraticKernel{eltype(t),Tr,T}(t,α)
end

@inline kappa(κ::RationalQuadraticKernel, d²::T) where {T<:Real} = (one(T)+d²/κ.α)^(-κ.α)


"""
    GammaRationalQuadraticKernel([ρ=1.0[,α=2.0[,γ=2.0]]])

    The Gamma-rational-quadratic kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y)=(1+||x−y||^(2γ)/α)^(-α)
```
    where α is a shape parameter of the Euclidean distance and γ is another shape parameter.
"""
struct GammaRationalQuadraticKernel{T,Tr,Tα<:Real,Tγ<:Real} <: Kernel{T,Tr}
    transform::Tr
    metric::SqEuclidean
    α::Tα
    γ::Tγ
    function GammaRationalQuadraticKernel{T,Tr,Tα,Tγ}(t::Tr,α::Tα,γ::Tγ) where {T,Tr,Tα<:Real,Tγ<:Real}
        new{T,Tr,Tα,Tγ}(t,SqEuclidean(),α,γ)
    end
end

function GammaRationalQuadraticKernel(ρ::T₁=1.0,α::T₂=2.0,γ::T₃=2.0) where {T₁<:Real,T₂<:Real,T₃<:Real}
    @check_args(GammaRationalQuadraticKernel, α, α > one(T₂), "α > 1")
    @check_args(GammaRationalQuadraticKernel, γ, γ >= one(T₂), "γ >= 1")
    GammaRationalQuadraticKernel{T₁,ScaleTransform{T₁},T₂,T₃}(ScaleTransform(ρ),α,γ)
end

function GammaRationalQuadraticKernel(ρ::A,α::T₁=2.0,γ::T₂=2.0) where {A<:AbstractVector{<:Real},T₁<:Real,T₂<:Real}
    @check_args(GammaRationalQuadraticKernel, α, α > one(T₁), "α > 1")
    @check_args(GammaRationalQuadraticKernel, γ, γ >= one(T₂), "γ >= 1")
    GammaRationalQuadraticKernel{eltype(A),ScaleTransform{A},T₁,T₂}(ScaleTransform(ρ),α,γ)
end

function GammaRationalQuadraticKernel(t::Tr,α::T₁=2.0,γ::T₂=2.0) where {Tr<:Transform,T₁<:Real,T₂<:Real}
    @check_args(GammaRationalQuadraticKernel, α, α > one(T₁), "α > 1")
    @check_args(GammaRationalQuadraticKernel, γ, γ >= one(T₂), "γ >= 1")
    GammaRationalQuadraticKernel{eltype(t),Tr,T₁,T₂}(t,α,γ)
end

@inline kappa(κ::GammaRationalQuadraticKernel, d²::T) where {T<:Real} = (one(T)+d²^κ.γ/κ.α)^(-κ.α)
