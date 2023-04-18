"""
    SqExponentialKernel(; metric=Euclidean())

Squared exponential kernel with respect to the `metric`.

# Definition

For inputs ``x, x'`` and metric ``d(\\cdot, \\cdot)``, the squared exponential kernel is
defined as
```math
k(x, x') = \\exp\\bigg(- \\frac{d(x, x')^2}{2}\\bigg).
```
By default, ``d`` is the Euclidean metric ``d(x, x') = \\|x - x'\\|_2``.

See also: [`GammaExponentialKernel`](@ref)
"""
struct SqExponentialKernel{M} <: SimpleKernel
    metric::M
end

SqExponentialKernel(; metric=Euclidean()) = SqExponentialKernel(metric)

kappa(::SqExponentialKernel, d::Real) = exp(-d^2 / 2)
kappa(::SqExponentialKernel{<:Euclidean}, d²::Real) = exp(-d² / 2)

metric(k::SqExponentialKernel) = k.metric
metric(::SqExponentialKernel{<:Euclidean}) = SqEuclidean()

iskroncompatible(::SqExponentialKernel) = true

function Base.show(io::IO, k::SqExponentialKernel)
    return print(io, "Squared Exponential Kernel (metric = ", k.metric, ")")
end

## Aliases ##

"""
    RBFKernel()

Alias of [`SqExponentialKernel`](@ref).
"""
const RBFKernel = SqExponentialKernel

"""
    GaussianKernel()

Alias of [`SqExponentialKernel`](@ref).
"""
const GaussianKernel = SqExponentialKernel

"""
    SEKernel()

Alias of [`SqExponentialKernel`](@ref).
"""
const SEKernel = SqExponentialKernel

"""
    ExponentialKernel(; metric=Euclidean())

Exponential kernel with respect to the `metric`.

# Definition

For inputs ``x, x'`` and metric ``d(\\cdot, \\cdot)``, the exponential kernel is defined as
```math
k(x, x') = \\exp\\big(- d(x, x')\\big).
```
By default, ``d`` is the Euclidean metric ``d(x, x') = \\|x - x'\\|_2``.

See also: [`GammaExponentialKernel`](@ref)
"""
struct ExponentialKernel{M} <: SimpleKernel
    metric::M
end

ExponentialKernel(; metric=Euclidean()) = ExponentialKernel(metric)

kappa(::ExponentialKernel, d::Real) = exp(-d)

metric(k::ExponentialKernel) = k.metric

iskroncompatible(::ExponentialKernel) = true

function Base.show(io::IO, k::ExponentialKernel)
    return print(io, "Exponential Kernel (metric = ", k.metric, ")")
end

## Aliases ##

"""
    LaplacianKernel()

Alias of [`ExponentialKernel`](@ref).
"""
const LaplacianKernel = ExponentialKernel

"""
    Matern12Kernel()

Alias of [`ExponentialKernel`](@ref).
"""
const Matern12Kernel = ExponentialKernel

"""
    GammaExponentialKernel(; γ::Real=1.0, metric=Euclidean())

γ-exponential kernel with respect to the `metric` and with parameter `γ`.

# Definition

For inputs ``x, x'`` and metric ``d(\\cdot, \\cdot)``, the γ-exponential kernel[^RW] with
parameter ``\\gamma \\in (0, 2]``
is defined as
```math
k(x, x'; \\gamma) = \\exp\\big(- d(x, x')^{\\gamma}\\big).
```
By default, ``d`` is the Euclidean metric ``d(x, x') = \\|x - x'\\|_2``.

See also: [`ExponentialKernel`](@ref), [`SqExponentialKernel`](@ref)

[^RW]: C. E. Rasmussen & C. K. I. Williams (2006). Gaussian Processes for Machine Learning.
"""
struct GammaExponentialKernel{Tγ<:Real,M} <: SimpleKernel
    γ::Vector{Tγ}
    metric::M

    function GammaExponentialKernel(γ::Real, metric; check_args::Bool=true)
        check_args && @check_args(GammaExponentialKernel, γ, zero(γ) < γ ≤ 2, "γ ∈ (0, 2]")
        return new{typeof(γ),typeof(metric)}([γ], metric)
    end
end

function GammaExponentialKernel(;
    gamma::Real=1.0, γ::Real=gamma, metric=Euclidean(), check_args::Bool=true
)
    return GammaExponentialKernel(γ, metric; check_args)
end

@functor GammaExponentialKernel

kappa(κ::GammaExponentialKernel, d::Real) = exp(-d^only(κ.γ))

metric(k::GammaExponentialKernel) = k.metric

iskroncompatible(::GammaExponentialKernel) = true

function Base.show(io::IO, κ::GammaExponentialKernel)
    return print(
        io, "Gamma Exponential Kernel (γ = ", only(κ.γ), ", metric = ", κ.metric, ")"
    )
end
