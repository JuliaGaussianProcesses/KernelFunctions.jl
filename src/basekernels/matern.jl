"""
    MaternKernel(; ν::Real=1.5)

Matérn kernel of order `ν`.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the Matérn kernel of order ``\\nu > 0`` is
defined as
```math
k(x,x';\\nu) = \\frac{2^{1-\\nu}}{\\Gamma(\\nu)}\\big(\\sqrt{2\\nu}\\|x-x'\\|_2\\big) K_\\nu\\big(\\sqrt{2\\nu}\\|x-x'\\|_2\\big),
```
where ``\\Gamma`` is the Gamma function and ``K_{\\nu}`` is the modified Bessel function of
the second kind of order ``\\nu``.

A Gaussian process with a Matérn kernel is ``\\lceil \\nu \\rceil - 1``-times
differentiable in the mean-square sense.

See also: [`Matern12Kernel`](@ref), [`Matern32Kernel`](@ref), [`Matern52Kernel`](@ref)
"""
struct MaternKernel{Tν<:Real} <: SimpleKernel
    ν::Vector{Tν}
    function MaternKernel(; nu::Real=1.5, ν::Real=nu)
        @check_args(MaternKernel, ν, ν > zero(ν), "ν > 0")
        return new{typeof(ν)}([ν])
    end
end

@functor MaternKernel

@inline function kappa(κ::MaternKernel, d::Real)
    result = _matern(first(κ.ν), d)
    return ifelse(iszero(d), one(result), result)
end

function _matern(ν::Real, d::Real)
    y = sqrt(2ν) * d
    return exp((one(d) - ν) * logtwo - loggamma(ν) + ν * log(y) + log(besselk(ν, y)))
end

binary_op(::MaternKernel) = Euclidean()

Base.show(io::IO, κ::MaternKernel) = print(io, "Matern Kernel (ν = ", first(κ.ν), ")")

## Matern12Kernel = ExponentialKernel aliased in exponential.jl

"""
    Matern32Kernel()

Matérn kernel of order ``3/2``.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the Matérn kernel of order ``3/2`` is given by
```math
k(x, x') = \\big(1 + \\sqrt{3} \\|x - x'\\|_2 \\big) \\exp\\big(- \\sqrt{3}\\|x - x'\\|_2\\big).
```

See also: [`MaternKernel`](@ref)
"""
struct Matern32Kernel <: SimpleKernel end

kappa(::Matern32Kernel, d::Real) = (1 + sqrt(3) * d) * exp(-sqrt(3) * d)

binary_op(::Matern32Kernel) = Euclidean()

Base.show(io::IO, ::Matern32Kernel) = print(io, "Matern 3/2 Kernel")

"""
    Matern52Kernel()

Matérn kernel of order ``5/2``.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the Matérn kernel of order ``5/2`` is given by
```math
k(x, x') = \\bigg(1 + \\sqrt{5} \\|x - x'\\|_2 + \\frac{5}{3}\\|x - x'\\|_2^2\\bigg)
           \\exp\\big(- \\sqrt{5}\\|x - x'\\|_2\\big).
```

See also: [`MaternKernel`](@ref)
"""
struct Matern52Kernel <: SimpleKernel end

kappa(::Matern52Kernel, d::Real) = (1 + sqrt(5) * d + 5 * d^2 / 3) * exp(-sqrt(5) * d)

binary_op(::Matern52Kernel) = Euclidean()

Base.show(io::IO, ::Matern52Kernel) = print(io, "Matern 5/2 Kernel")
