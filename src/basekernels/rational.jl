"""
    RationalKernel(; α::Real=2.0, metric=Euclidean())

Rational kernel with shape parameter `α` and given `metric`.

# Definition

For inputs ``x, x'`` and metric ``d(\\cdot, \\cdot)``, the rational kernel with shape parameter ``\\alpha > 0`` is defined as
```math
k(x, x'; \\alpha) = \\bigg(1 + \\frac{d(x, x')}{\\alpha}\\bigg)^{-\\alpha}.
```
By default, ``d`` is the Euclidean metric ``d(x, x') = \\|x - x'\\|_2``.

The [`ExponentialKernel`](@ref) is recovered in the limit as ``\\alpha \\to \\infty``.

See also: [`GammaRationalKernel`](@ref)
"""
struct RationalKernel{Tα<:Real,M} <: SimpleKernel
    α::Vector{Tα}
    metric::M

    function RationalKernel(α::Real, metric)
        @check_args(RationalKernel, α, α > zero(α), "α > 0")
        return new{typeof(α),typeof(metric)}([α], metric)
    end
end

function RationalKernel(; alpha::Real=2.0, α::Real=alpha, metric=Euclidean())
    return RationalKernel(α, metric)
end

@functor RationalKernel

function kappa(κ::RationalKernel, d::Real)
    return (one(d) + d / only(κ.α))^(-only(κ.α))
end

metric(k::RationalKernel) = k.metric

function Base.show(io::IO, κ::RationalKernel)
    return print(io, "Rational Kernel (α = ", only(κ.α), ", metric = ", κ.metric, ")")
end

"""
    RationalQuadraticKernel(; α::Real=2.0, metric=Euclidean())

Rational-quadratic kernel with respect to the `metric` and with shape parameter `α`.

# Definition

For inputs ``x, x'`` and metric ``d(\\cdot, \\cdot)``, the rational-quadratic kernel with
shape parameter ``\\alpha > 0`` is defined as
```math
k(x, x'; \\alpha) = \\bigg(1 + \\frac{d(x, x')^2}{2\\alpha}\\bigg)^{-\\alpha}.
```
By default, ``d`` is the Euclidean metric ``d(x, x') = \\|x - x'\\|_2``.

The [`SqExponentialKernel`](@ref) is recovered in the limit as ``\\alpha \\to \\infty``.

See also: [`GammaRationalKernel`](@ref)
"""
struct RationalQuadraticKernel{Tα<:Real,M} <: SimpleKernel
    α::Vector{Tα}
    metric::M

    function RationalQuadraticKernel(; alpha::Real=2.0, α::Real=alpha, metric=Euclidean())
        @check_args(RationalQuadraticKernel, α, α > zero(α), "α > 0")
        return new{typeof(α),typeof(metric)}([α], metric)
    end
end

@functor RationalQuadraticKernel

function kappa(κ::RationalQuadraticKernel, d::Real)
    return (one(d) + d^2 / (2 * only(κ.α)))^(-only(κ.α))
end
function kappa(κ::RationalQuadraticKernel{<:Real,<:Euclidean}, d²::Real)
    return (one(d²) + d² / (2 * only(κ.α)))^(-only(κ.α))
end

metric(k::RationalQuadraticKernel) = k.metric
metric(::RationalQuadraticKernel{<:Real,<:Euclidean}) = SqEuclidean()

function Base.show(io::IO, κ::RationalQuadraticKernel)
    return print(
        io, "Rational Quadratic Kernel (α = ", only(κ.α), ", metric = ", κ.metric, ")"
    )
end

"""
    GammaRationalKernel(; α::Real=2.0, γ::Real=1.0, metric=Euclidean())

γ-rational kernel with respect to the `metric` with shape parameters `α` and `γ`.

# Definition

For inputs ``x, x'`` and metric ``d(\\cdot, \\cdot)``, the γ-rational kernel with shape
parameters ``\\alpha > 0`` and ``\\gamma \\in (0, 2]`` is defined as
```math
k(x, x'; \\alpha, \\gamma) = \\bigg(1 + \\frac{d(x, x')^{\\gamma}}{\\alpha}\\bigg)^{-\\alpha}.
```
By default, ``d`` is the Euclidean metric ``d(x, x') = \\|x - x'\\|_2``.

The [`GammaExponentialKernel`](@ref) is recovered in the limit as ``\\alpha \\to \\infty``.

See also: [`RationalKernel`](@ref), [`RationalQuadraticKernel`](@ref)
"""
struct GammaRationalKernel{Tα<:Real,Tγ<:Real,M} <: SimpleKernel
    α::Vector{Tα}
    γ::Vector{Tγ}
    metric::M

    function GammaRationalKernel(;
        alpha::Real=2.0, gamma::Real=1.0, α::Real=alpha, γ::Real=gamma, metric=Euclidean()
    )
        @check_args(GammaRationalKernel, α, α > zero(α), "α > 0")
        @check_args(GammaRationalKernel, γ, zero(γ) < γ ≤ 2, "γ ∈ (0, 2]")
        return new{typeof(α),typeof(γ),typeof(metric)}([α], [γ], metric)
    end
end

@functor GammaRationalKernel

function kappa(κ::GammaRationalKernel, d::Real)
    return (one(d) + d^only(κ.γ) / only(κ.α))^(-only(κ.α))
end

metric(k::GammaRationalKernel) = k.metric

function Base.show(io::IO, κ::GammaRationalKernel)
    return print(
        io,
        "Gamma Rational Kernel (α = ",
        only(κ.α),
        ", γ = ",
        only(κ.γ),
        ", metric = ",
        κ.metric,
        ")",
    )
end

@doc raw"""
    InverseMultiQuadricKernel(; α::Real=1.0, c::Real=1.0, metric=Euclidean())

Inverse multiquadric kernel with respect to the `metric` with parameters `α` and `c`.

# Definition

For inputs ``x, x'`` and metric ``d(\cdot, \cdot)``, the inverse multiquadric kernel with
parameters ``\alpha, c > 0`` is defined as
```math
k(x, x'; \alpha, c) = \big(c + d(x, x')^2\big)^{-\alpha}.
```
By default, ``d`` is the Euclidean metric ``d(x, x') = \|x - x'\|_2``.

For ``\alpha = c = 1``, the [`GammaRationalKernel`](@ref) with parameters ``\alpha = 1``
and ``\gamma = 2`` is recovered.

For ``\alpha = 1/2`` and ``c = 1``, the [`RationalQuadraticKernel`](@ref) with parameter
``\alpha = 1/2`` is recovered.

# References

Micchelli, C.A. (1986). Interpolation of scattered data: Distance matrices and conditionally
positive definite functions. Constructive Approximation 2, 11-22.
"""
struct InverseMultiQuadricKernel{Tα<:Real,Tc<:Real,M} <: SimpleKernel
    α::Vector{Tα}
    c::Vector{Tc}
    metric::M

    function InverseMultiQuadricKernel(α::Real, c::Real, metric)
        @check_args(InverseMultiQuadricKernel, α, α > zero(α), "α > 0")
        @check_args(InverseMultiQuadricKernel, c, c > zero(c), "c > 0")
        return new{typeof(α),typeof(c),typeof(metric)}([α], [c], metric)
    end
end

function InverseMultiQuadricKernel(;
    alpha::Real=1.0, α::Real=alpha, c::Real=1.0, metric=Euclidean()
)
    return InverseMultiQuadricKernel(α, c, metric)
end

@functor InverseMultiQuadricKernel

function kappa(k::InverseMultiQuadricKernel, d::Real)
    return (first(k.c) + d^2)^(-first(k.α))
end
function kappa(k::InverseMultiQuadricKernel{<:Real,<:Real,<:Euclidean}, d2::Real)
    return (first(k.c) + d2)^(-first(k.α))
end

metric(k::InverseMultiQuadricKernel) = k.metric
metric(::InverseMultiQuadricKernel{<:Real,<:Real,<:Euclidean}) = SqEuclidean()

function Base.show(io::IO, k::InverseMultiQuadricKernel)
    return print(
        io,
        "Inverse Multiquadric Kernel (α = ",
        first(k.α),
        ", c = ",
        first(k.c),
        ", metric = ",
        k.metric,
        ")",
    )
end
