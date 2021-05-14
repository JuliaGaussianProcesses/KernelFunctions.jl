"""
    MaternKernel(; ν::Real=1.5, metric=Euclidean())

Matérn kernel of order `ν` with respect to the `metric`.

# Definition

For inputs ``x, x'`` and metric ``d(\\cdot, \\cdot)``, the Matérn kernel of order
``\\nu > 0`` is defined as
```math
k(x,x';\\nu) = \\frac{2^{1-\\nu}}{\\Gamma(\\nu)}\\big(\\sqrt{2\\nu} d(x, x')\\big) K_\\nu\\big(\\sqrt{2\\nu} d(x, x')\\big),
```
where ``\\Gamma`` is the Gamma function and ``K_{\\nu}`` is the modified Bessel function of
the second kind of order ``\\nu``.
By default, ``d`` is the Euclidean metric ``d(x, x') = \|x - x'\|_2``.

A Gaussian process with a Matérn kernel is ``\\lceil \\nu \\rceil - 1``-times
differentiable in the mean-square sense.

See also: [`Matern12Kernel`](@ref), [`Matern32Kernel`](@ref), [`Matern52Kernel`](@ref)
"""
struct MaternKernel{Tν<:Real,M} <: SimpleKernel
    ν::Vector{Tν}
    metric::M

    function MaternKernel(; nu::Real=1.5, ν::Real=nu, metric=Euclidean())
        @check_args(MaternKernel, ν, ν > zero(ν), "ν > 0")
        return new{typeof(ν),typeof(metric)}([ν], metric)
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

metric(k::MaternKernel) = k.metric

function Base.show(io::IO, κ::MaternKernel)
    return print(io, "Matern Kernel (ν = ", first(κ.ν), ", metric = ", κ.metric, ")")
end

## Matern12Kernel = ExponentialKernel aliased in exponential.jl

"""
    Matern32Kernel(; metric=Euclidean())

Matérn kernel of order ``3/2`` with respect to the `metric`.

# Definition

For inputs ``x, x'`` and metric ``d(\\cdot, \\cdot)``, the Matérn kernel of order ``3/2`` is
 given by
```math
k(x, x') = \\big(1 + \\sqrt{3} d(x, x') \\big) \\exp\\big(- \\sqrt{3} d(x, x') \\big).
```
By default, ``d`` is the Euclidean metric ``d(x, x') = \|x - x'\|_2``.

See also: [`MaternKernel`](@ref)
"""
struct Matern32Kernel{M} <: SimpleKernel
    metric::M

    function Matern32Kernel(; metric=Euclidean())
        return new{typeof(metric)}(metric)
    end
end

kappa(::Matern32Kernel, d::Real) = (1 + sqrt(3) * d) * exp(-sqrt(3) * d)

metric(k::Matern32Kernel) = k.metric

function Base.show(io::IO, k::Matern32Kernel)
    return print(io, "Matern 3/2 Kernel (metric = ", k.metric, ")")
end

"""
    Matern52Kernel(; metric=Euclidean())

Matérn kernel of order ``5/2`` with respect to the `metric`.

# Definition

For inputs ``x, x'`` and metric ``d(\\cdot, \\cdot)``, the Matérn kernel of order ``5/2`` is
given by
```math
k(x, x') = \\bigg(1 + \\sqrt{5} d(x, x') + \\frac{5}{3} d(x, x')^2\\bigg)
           \\exp\\big(- \\sqrt{5} d(x, x') \\big).
```
By default, ``d`` is the Euclidean metric ``d(x, x') = \|x - x'\|_2``.

See also: [`MaternKernel`](@ref)
"""
struct Matern52Kernel{M} <: SimpleKernel
    metric::M

    function Matern52Kernel(; metric=Euclidean())
        return new{typeof(metric)}(metric)
    end
end

kappa(::Matern52Kernel, d::Real) = (1 + sqrt(5) * d + 5 * d^2 / 3) * exp(-sqrt(5) * d)

metric(k::Matern52Kernel) = k.metric

function Base.show(io::IO, k::Matern52Kernel)
    return print(io, "Matern 5/2 Kernel (metric = ", k.metric, ")")
end
