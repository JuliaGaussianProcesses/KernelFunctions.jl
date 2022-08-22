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

    function LinearKernel(c::Real)
        @check_args(LinearKernel, c, c >= zero(c), "c ≥ 0")
        return new{typeof(c)}([c])
    end
end

LinearKernel(; c::Real=0.0) = LinearKernel(c)

@functor LinearKernel

__linear_kappa(c::Real, xᵀy::Real) = xᵀy + c

kappa(κ::LinearKernel, xᵀy::Real) = __linear_kappa(only(κ.c), xᵀy)

metric(::LinearKernel) = DotProduct()

function kernelmatrix(k::LinearKernel, x::AbstractVector, y::AbstractVector)
    return __linear_kappa.(only(k.c), pairwise(metric(k), x, y))
end

function kernelmatrix(k::LinearKernel, x::AbstractVector)
    return __linear_kappa.(only(k.c), pairwise(metric(k), x))
end

function kernelmatrix_diag(k::LinearKernel, x::AbstractVector, y::AbstractVector)
    return __linear_kappa.(only(k.c), colwise(metric(k), x, y))
end

function kernelmatrix_diag(k::LinearKernel, x::AbstractVector)
    return __linear_kappa.(only(k.c), colwise(metric(k), x))
end

Base.show(io::IO, κ::LinearKernel) = print(io, "Linear Kernel (c = ", only(κ.c), ")")

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
        @check_args(PolynomialKernel, c, only(c) >= zero(Tc), "c ≥ 0")
        return new{Tc}(degree, c)
    end
end

function PolynomialKernel(; degree::Int=2, c::Real=0.0)
    return PolynomialKernel{typeof(c)}(degree, [c])
end

# The degree of the polynomial kernel is a fixed discrete parameter
function Functors.functor(::Type{<:PolynomialKernel}, x)
    reconstruct_polynomialkernel(xs) = PolynomialKernel{typeof(xs.c)}(x.degree, xs.c)
    return (c=x.c,), reconstruct_polynomialkernel
end

@noinline __make_polynomial_kappa(degree) = (c::Real, xᵀy::Real) -> (xᵀy + c)^degree

kappa(κ::PolynomialKernel, xᵀy::Real) = __make_polynomial_kappa(κ.degree)(only(κ.c), xᵀy)

metric(::PolynomialKernel) = DotProduct()

function kernelmatrix(k::PolynomialKernel, x::AbstractVector, y::AbstractVector)
    return __make_polynomial_kappa(k.degree).(only(k.c), pairwise(metric(k), x, y))
end

function kernelmatrix(k::PolynomialKernel, x::AbstractVector)
    return __make_polynomial_kappa(k.degree).(only(k.c), pairwise(metric(k), x))
end

function kernelmatrix_diag(k::PolynomialKernel, x::AbstractVector, y::AbstractVector)
    return __make_polynomial_kappa(k.degree).(only(k.c), colwise(metric(k), x, y))
end

function kernelmatrix_diag(k::PolynomialKernel, x::AbstractVector)
    return __make_polynomial_kappa(k.degree).(only(k.c), colwise(metric(k), x))
end

function Base.show(io::IO, κ::PolynomialKernel)
    return print(io, "Polynomial Kernel (c = ", only(κ.c), ", degree = ", κ.degree, ")")
end
