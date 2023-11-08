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
struct LinearKernel{T<:Real} <: SimpleKernel
    c::T

    function LinearKernel(c::Real)
        @check_args(LinearKernel, c, c >= zero(c), "c ≥ 0")
        return new{typeof(c)}(c)
    end
end

LinearKernel(; c::Real=0.0) = LinearKernel(c)

function ParameterHandling.flatten(::Type{T}, k::LinearKernel{S}) where {T<:Real,S<:Real}
    function unflatten_to_linearkernel(v::Vector{T})
        return LinearKernel(S(exp(only(v))))
    end
    return T[log(k.c)], unflatten_to_linearkernel
end

__linear_kappa(c::Real, xᵀy::Real) = xᵀy + c

kappa(κ::LinearKernel, xᵀy::Real) = __linear_kappa(κ.c, xᵀy)

metric(::LinearKernel) = DotProduct()

function kernelmatrix(k::LinearKernel, x::AbstractVector, y::AbstractVector)
    return __linear_kappa.(k.c, pairwise(metric(k), x, y))
end

function kernelmatrix(k::LinearKernel, x::AbstractVector)
    return __linear_kappa.(k.c, pairwise(metric(k), x))
end

function kernelmatrix_diag(k::LinearKernel, x::AbstractVector, y::AbstractVector)
    return __linear_kappa.(k.c, colwise(metric(k), x, y))
end

function kernelmatrix_diag(k::LinearKernel, x::AbstractVector)
    return __linear_kappa.(k.c, colwise(metric(k), x))
end

Base.show(io::IO, κ::LinearKernel) = print(io, "Linear Kernel (c = ", κ.c, ")")

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
struct PolynomialKernel{T<:Real} <: SimpleKernel
    degree::Int
    c::T

    function PolynomialKernel(degree::Int, c::Real)
        @check_args(PolynomialKernel, degree, degree >= one(degree), "degree ≥ 1")
        @check_args(PolynomialKernel, c, c >= zero(c), "c ≥ 0")
        return new{typeof(c)}(degree, c)
    end
end

PolynomialKernel(; degree::Int=2, c::Real=0.0) = PolynomialKernel(degree, c)

function ParameterHandling.flatten(
    ::Type{T}, k::PolynomialKernel{S}
) where {T<:Real,S<:Real}
    degree = k.degree
    function unflatten_to_polynomialkernel(v::Vector{T})
        return PolynomialKernel(degree, S(exp(only(v))))
    end
    return T[log(k.c)], unflatten_to_polynomialkernel
end

# The degree of the polynomial kernel is a fixed discrete parameter
function Functors.functor(::Type{<:PolynomialKernel}, x)
    reconstruct_polynomialkernel(xs) = PolynomialKernel{typeof(xs.c)}(x.degree, xs.c)
    return (c=x.c,), reconstruct_polynomialkernel
end

struct _PolynomialKappa
    degree::Int
end

(κ::_PolynomialKappa)(c::Real, xᵀy::Real) = (xᵀy + c)^κ.degree

kappa(κ::PolynomialKernel, xᵀy::Real) = _PolynomialKappa(κ.degree)(κ.c, xᵀy)

metric(::PolynomialKernel) = DotProduct()

function kernelmatrix(k::PolynomialKernel, x::AbstractVector, y::AbstractVector)
    return _PolynomialKappa(k.degree).(k.c, pairwise(metric(k), x, y))
end

function kernelmatrix(k::PolynomialKernel, x::AbstractVector)
    return _PolynomialKappa(k.degree).(k.c, pairwise(metric(k), x))
end

function kernelmatrix_diag(k::PolynomialKernel, x::AbstractVector, y::AbstractVector)
    return _PolynomialKappa(k.degree).(k.c, colwise(metric(k), x, y))
end

function kernelmatrix_diag(k::PolynomialKernel, x::AbstractVector)
    return _PolynomialKappa(k.degree).(k.c, colwise(metric(k), x))
end

function Base.show(io::IO, κ::PolynomialKernel)
    return print(io, "Polynomial Kernel (c = ", κ.c, ", degree = ", κ.degree, ")")
end
