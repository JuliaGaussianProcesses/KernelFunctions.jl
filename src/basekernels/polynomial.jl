"""
    LinearKernel(; c::Real=0)

Linear kernel with constant offset `c`.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^k``, the linear kernel with constant offset
``c \\in \\mathbb{R}`` is defined as
```math
k(x, x'; c) = x^\\top x' + c.
```

See also: [`PolynomialKernel`](@ref)
"""
struct LinearKernel{Tc<:Real} <: SimpleKernel
    c::Vector{Tc}
    function LinearKernel(; c::Real=0)
        return new{typeof(c)}([c])
    end
end

@functor LinearKernel

kappa(κ::LinearKernel, xᵀy::Real) = xᵀy + first(κ.c)

metric(::LinearKernel) = DotProduct()

Base.show(io::IO, κ::LinearKernel) = print(io, "Linear Kernel (c = ", first(κ.c), ")")

"""
    PolynomialKernel(; d::Real=2, c::Real=0)

Polynomial kernel of degree `d` with constant offset `c`.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^k``, the polynomial kernel of degree ``d \\geq 1``
with constant offset ``c \\in \\mathbb{R}`` is defined as
```math
k(x, x'; c, d) = (x^\\top x' + c)^d.
```

See also: [`LinearKernel`](@ref)
"""
struct PolynomialKernel{Td<:Real,Tc<:Real} <: SimpleKernel
    d::Vector{Td}
    c::Vector{Tc}
    function PolynomialKernel(; d::Real=2, c::Real=0)
        @check_args(PolynomialKernel, d, d >= one(d), "d >= 1")
        return new{typeof(d),typeof(c)}([d], [c])
    end
end

@functor PolynomialKernel

kappa(κ::PolynomialKernel, xᵀy::Real) = (xᵀy + first(κ.c))^(first(κ.d))

metric(::PolynomialKernel) = DotProduct()

function Base.show(io::IO, κ::PolynomialKernel)
    return print(io, "Polynomial Kernel (c = ", first(κ.c), ", d = ", first(κ.d), ")")
end
