struct LinearKernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::DotProduct
    c::T
    function LinearKernel{T,Tr}(transform::Tr,c) where {T,Tr<:Transform}
        return new{T,Tr}(transform,DotProduct(),c)
    end
end

@inline kappa(κ::LinearKernel, xᵀy::T) where {T<:Real} = xᵀy + κ.c

struct PolynomialKernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::DotProduct
    c::T
    d::Int
    function PolynomialKernel{T,Tr}(transform::Tr,c) where {T,Tr<:Transform}
        return new{T,Tr}(transform,DotProduct(),c,d)
    end
end

@inline kappa(κ::LinearKernel, xᵀy::T) where {T<:Real} = (xᵀy + κ.c)^(κ.d)
