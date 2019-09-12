struct LinearKernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::DotProduct
    c::T
    function LinearKernel{T,Tr}(transform::Tr,c::T) where {T,Tr<:Transform}
        return new{T,Tr}(transform,DotProduct(),c)
    end
end

function LinearKernel(ρ::T₁=1.0,c::T₂=zero(T₁)) where {T₁<:Real,T₂<:Real}
    LinearKernel{T₁,ScaleTransform{T₁}}(ScaleTransform(ρ),c)
end

function LinearKernel(ρ::A,c::T=zero(eltype(ρ))) where {A<:AbstractVector{<:Real},T<:Real}
    LinearKernel{eltype(A),ScaleTransform{A}}(ScaleTransform(ρ),eltype(A)(c))
end

function LinearKernel(t::Tr,c::T=zero(eltype(t))) where {Tr<:Transform,T<:Real}
    LinearKernel{eltype(t),Tr}(t,eltype(Tr)(c))
end

@inline kappa(κ::LinearKernel, xᵀy::T) where {T<:Real} = xᵀy + κ.c

struct PolynomialKernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::DotProduct
    c::T
    d::Real
    function PolynomialKernel{T,Tr}(transform::Tr,c::T,d::Real) where {T,Tr<:Transform}
        return new{T,Tr}(transform,DotProduct(),c,d)
    end
end

function PolynomialKernel(ρ::T₁=1.0,d::T₂=2.0,c::T₃=zero(T₁)) where {T₁<:Real,T₂<:Real,T₃<:Real}
    @check_args(PolynomialKernel, d, d >= one(T₁), "d >= 1")
    Polynomial{T₁,ScaleTransform{T₁}}(ScaleTransform(ρ),T₁(c),d)
end

function PolynomialKernel(ρ::A,d::T₁=2.0,c::T₂=zero(eltype(ρ))) where {A<:AbstractVector{<:Real},T₁<:Real,T₂<:Real}
    @check_args(PolynomialKernel, d, d >= one(T₁), "d >= 1")
    PolynomialKernel{eltype(A),ScaleTransform{A}}(ScaleTransform(ρ),eltype(A)(c),d)
end

function PolynomialKernel(t::Tr,d::T₁=2.0,c::T₂=zero(eltype(t))) where {Tr<:Transform,T₁<:Real,T₂<:Real}
    @check_args(PolynomialKernel, d, d >= one(T₁), "d >= 1")
    PolynomialKernel{eltype(Tr),Tr}(t,eltype(Tr)(c),d)
end

@inline kappa(κ::PolynomialKernel, xᵀy::T) where {T<:Real} = (xᵀy + κ.c)^(κ.d)
