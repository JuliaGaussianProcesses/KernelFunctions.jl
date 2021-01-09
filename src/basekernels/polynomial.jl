"""
    LinearKernel(; c = 0.0)

The linear kernel is a Mercer kernel given by
```
    κ(x,y) = xᵀy + c
```
Where `c` is a real number
"""
struct LinearKernel{Tc<:Real} <: SimpleKernel
    c::Vector{Tc}
    function LinearKernel(; c::T=0.0) where {T}
        return new{T}([c])
    end
end

@functor LinearKernel

kappa(κ::LinearKernel, xᵀy::Real) = xᵀy + first(κ.c)

metric(::LinearKernel) = DotProduct()

Base.show(io::IO, κ::LinearKernel) = print(io, "Linear Kernel (c = ", first(κ.c), ")")

"""
    PolynomialKernel(; d = 2.0, c = 0.0)

The polynomial kernel is a Mercer kernel given by
```
    κ(x,y) = (xᵀy + c)^d
```
Where `c` is a real number, and `d` is a shape parameter bigger than 1. For `d = 1` see [`LinearKernel`](@ref)
"""
struct PolynomialKernel{Td<:Real,Tc<:Real} <: SimpleKernel
    d::Vector{Td}
    c::Vector{Tc}
    function PolynomialKernel(; d::Td=2.0, c::Tc=0.0) where {Td<:Real,Tc<:Real}
        @check_args(PolynomialKernel, d, d >= one(Td), "d >= 1")
        return new{Td,Tc}([d], [c])
    end
end

@functor PolynomialKernel

kappa(κ::PolynomialKernel, xᵀy::Real) = (xᵀy + first(κ.c))^(first(κ.d))

metric(::PolynomialKernel) = DotProduct()

function Base.show(io::IO, κ::PolynomialKernel)
    return print(io, "Polynomial Kernel (c = ", first(κ.c), ", d = ", first(κ.d), ")")
end
