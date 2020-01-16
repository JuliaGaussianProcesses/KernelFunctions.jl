"""
`LinearKernel([ρ=1.0,[c=0.0]])`
The linear kernel is a Mercer kernel given by
```
    κ(x,y) = ρ²xᵀy + c
```
Where `c` is a real number
"""
struct LinearKernel{Tr, Tc<:Real} <: Kernel{Tr}
    transform::Tr
    c::Tc
end

function LinearKernel(ρ::T=1.0, c::Real=zero(T)) where {T<:Real}
    LinearKernel(ScaleTransform(ρ), c)
end

function LinearKernel(ρ::AbstractVector{T}, c::Real=zero(T)) where {T<:Real}
    LinearKernel(ARDTransform(ρ), c)
end

LinearKernel(t::Transform) = LinearKernel(t, 0.0)

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
struct PolynomialKernel{Tr,Tc<:Real,Td<:Real} <: Kernel{Tr}
    transform::Tr
    d::Td
    c::Tc
    function PolynomialKernel{Tr, Tc, Td}(transform::Tr, d::Td, c::Tc) where {Tr<:Transform, Td<:Real, Tc<:Real}
        @check_args(PolynomialKernel, d, d >= one(Td), "d >= 1")
        return new{Tr, Td, Tc}(transform,d, c)
    end
end

function PolynomialKernel(ρ::Real=1.0, d::Td=2.0, c::Real=zero(Td)) where {Td<:Real}
    PolynomialKernel(ScaleTransform(ρ), d, c)
end

function PolynomialKernel(ρ::AbstractVector{T}, d::Real=2.0, c::Real=zero(T₁)) where {T<:Real}
    PolynomialKernel(ARDTransform(ρ), d, c)
end

function PolynomialKernel(t::Tr, d::Td=2.0, c::Tc=zero(eltype(Td))) where {Tr<:Transform, Td<:Real, Tc<:Real}
    PolynomialKernel{Tr, Tc, Td}(t, d, c)
end

params(k::PolynomialKernel) = (params(transform(k)),k.d,k.c)
opt_params(k::PolynomialKernel) = (opt_params(transform(k)),k.d,k.c)

@inline kappa(κ::PolynomialKernel, xᵀy::T) where {T<:Real} = (xᵀy + κ.c)^(κ.d)

metric(::PolynomialKernel) = DotProduct()
