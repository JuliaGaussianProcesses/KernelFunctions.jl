"""
    LinearKernel(; c::Real=0.0)

Linear kernel with constant offset `c`.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the linear kernel with constant offset
``c \\geq 0`` is defined as
```math
k(x, x'; c) = x^\\top x' + c.
```

See also: [`PolynomialKernel`](@ref)
"""
struct LinearKernel{Tc<:Real} <: SimpleKernel
    c::Vector{Tc}
    function LinearKernel(; c::Real=0.0)
        return new{typeof(c)}([c])
    end
end

@functor LinearKernel

kappa(κ::LinearKernel, xᵀy::Real) = xᵀy + first(κ.c)

metric(::LinearKernel) = DotProduct()

Base.show(io::IO, κ::LinearKernel) = print(io, "Linear Kernel (c = ", first(κ.c), ")")

"""
    PolynomialKernel(; degree::Int=2, c::Real=0.0)

Polynomial kernel of degree `degree` with constant offset `c`.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the polynomial kernel of degree
``\\nu \\in \\mathbb{N}`` with constant offset ``c \\geq 0`` is defined as
```math
k(x, x'; c, \\nu) = (x^\\top x' + c)^\\nu.
```

See also: [`LinearKernel`](@ref)
"""
struct PolynomialKernel{Tc<:Real} <: SimpleKernel
    degree::Int
    c::Vector{Tc}

    function PolynomialKernel{Tc}(degree::Int, c::Vector{Tc}) where {Tc}
        @check_args(PolynomialKernel, degree, degree >= one(degree), "degree ≥ 1")
        @check_args(PolynomialKernel, c, first(c) >= zero(Tc), "c ≥ 0")
        return new{Tc}(degree, c)
    end
end

function PolynomialKernel(; d::Real=-1, degree::Int=2, c::Real=0.0)
    if d != -1
        Base.depwarn(
            "keyword argument `d` is deprecated, use `degree` instead",
            :PiecewisePolynomialKernel,
        )
        isinteger(d) || error("polynomial degree has to be an integer")
        degree::Int = convert(Int, d)
    end
    return PolynomialKernel{typeof(c)}(degree, [c])
end

# The degree of the polynomial kernel is a fixed discrete parameter
function Functors.functor(::Type{<:PolynomialKernel}, x)
    reconstruct_polynomialkernel(xs) = PolynomialKernel{typeof(xs.c)}(x.degree, xs.c)
    return (c=x.c,), recconstruct_polynomialkernel
end

kappa(κ::PolynomialKernel, xᵀy::Real) = (xᵀy + first(κ.c))^κ.degree

metric(::PolynomialKernel) = DotProduct()

function Base.show(io::IO, κ::PolynomialKernel)
    return print(io, "Polynomial Kernel (c = ", first(κ.c), ", degree = ", κ.degree, ")")
end
