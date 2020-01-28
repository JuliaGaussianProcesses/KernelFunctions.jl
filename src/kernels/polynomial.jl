"""
`LinearKernel([ρ=1.0,[c=0.0]])`
The linear kernel is a Mercer kernel given by
```
    κ(x,y) = ρ²xᵀy + c
```
Where `c` is a real number
"""
struct LinearKernel{Tc<:Real} <: Kernel
    c::Tc
end

function LinearKernel(c::Tc=0.0) where {Tc}
    LinearKernel{Tc}(c)
end

params(k::LinearKernel) = (k.c)
opt_params(k::LinearKernel) = (k.c)

@inline kappa(κ::LinearKernel, xᵀy::Real) = xᵀy + κ.c

metric(::LinearKernel) = DotProduct()

"""
`PolynomialKernel([ρ=1.0[,d=2.0[,c=0.0]]])`
The polynomial kernel is a Mercer kernel given by
```
    κ(x,y) = (ρ²xᵀy + c)^d
```
Where `c` is a real number, and `d` is a shape parameter bigger than 1
"""
struct PolynomialKernel{Tc<:Real,Td<:Real} <: Kernel
    d::Td
    c::Tc
end

function PolynomialKernel(d::Td=2.0, c::Tc=zero(Td)) where {Td<:Real, Tc<:Real}
    @check_args(PolynomialKernel, d, d >= one(Td), "d >= 1")
    return PolynomialKernel{Td, Tc}(d, c)
end

params(k::PolynomialKernel) = (k.d,k.c)
opt_params(k::PolynomialKernel) = (k.d,k.c)

@inline kappa(κ::PolynomialKernel, xᵀy::T) where {T<:Real} = (xᵀy + κ.c)^(κ.d)

metric(::PolynomialKernel) = DotProduct()
