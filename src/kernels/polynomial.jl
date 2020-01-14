"""
`LinearKernel([ρ=1.0,[c=0.0]])`
The linear kernel is a Mercer kernel given by
```
    κ(x,y) = ρ²xᵀy + c
```
Where `c` is a real number
"""
struct LinearKernel{T,Tr,Tc<:Real} <: Kernel{T,Tr}
    transform::Tr
    c::Tc

    function LinearKernel{T,Tr,Tc}(transform::Tr,c::Tc) where {T,Tr<:Transform,Tc<:Real}
        return new{T,Tr,Tc}(transform,c)
    end
end

function LinearKernel(ρ::T₁=1.0,c::T₂=zero(T₁)) where {T₁<:Real,T₂<:Real}
    LinearKernel{T₁,ScaleTransform{T₁},T₂}(ScaleTransform(ρ),c)
end

function LinearKernel(ρ::AbstractVector{T₁},c::T₂=zero(T₁)) where {T₁<:Real,T₂<:Real}
    LinearKernel{T₁,ARDTransform{T₁,length(ρ)},T₂}(ARDTransform(ρ),c)
end

function LinearKernel(t::Tr,c::T=zero(Float64)) where {Tr<:Transform,T<:Real}
    LinearKernel{eltype(t),Tr,T}(t,c)
end

params(k::LinearKernel) = (params(transform(k)),k.c)
opt_params(k::LinearKernel) = (opt_params(transform(k)),k.c)

@inline kappa(κ::LinearKernel, xᵀy::T) where {T<:Real} = xᵀy + κ.c

metric(::LinearKernel) = DotProduct()

"""
`PolynomialKernel([ρ=1.0[,d=2.0[,c=0.0]]])`
The polynomial kernel is a Mercer kernel given by
```
    κ(x,y) = (ρ²xᵀy + c)^d
```
Where `c` is a real number, and `d` is a shape parameter bigger than 1
"""
struct PolynomialKernel{T,Tr,Tc<:Real,Td<:Real} <: Kernel{T,Tr}
    transform::Tr
    c::Tc
    d::Td

    function PolynomialKernel{T,Tr,Tc,Td}(transform::Tr,c::Tc,d::Td) where {T,Tr<:Transform,Tc<:Real,Td<:Real}
        return new{T,Tr,Tc,Td}(transform,c,d)
    end
end

function PolynomialKernel(ρ::T₁=1.0,d::T₂=2.0,c::T₃=zero(T₁)) where {T₁<:Real,T₂<:Real,T₃<:Real}
    @check_args(PolynomialKernel, d, d >= one(T₁), "d >= 1")
    PolynomialKernel{T₁,ScaleTransform{T₁},T₂,T₃}(ScaleTransform(ρ),c,d)
end

function PolynomialKernel(ρ::AbstractVector{T₁},d::T₂=2.0,c::T₃=zero(T₁)) where {T₁<:Real,T₂<:Real,T₃<:Real}
    @check_args(PolynomialKernel, d, d >= one(T₂), "d >= 1")
    PolynomialKernel{T₁,ARDTransform{T₁,length(ρ)},T₂,T₃}(ARDTransform(ρ),c,d)
end

function PolynomialKernel(t::Tr,d::T₁=2.0,c::T₂=zero(eltype(T₁))) where {Tr<:Transform,T₁<:Real,T₂<:Real}
    @check_args(PolynomialKernel, d, d >= one(T₁), "d >= 1")
    PolynomialKernel{eltype(Tr),Tr,T₁,T₂}(t,c,d)
end

params(k::PolynomialKernel) = (params(transform(k)),k.d,k.c)
opt_params(k::PolynomialKernel) = (opt_params(transform(k)),k.d,k.c)

@inline kappa(κ::PolynomialKernel, xᵀy::T) where {T<:Real} = (xᵀy + κ.c)^(κ.d)

metric(::PolynomialKernel) = DotProduct()