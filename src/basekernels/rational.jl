"""
    RationalKernel(; α::Real=2.0, metric=Euclidean())

Rational kernel with shape parameter `α` and given `metric`.

# Definition

For inputs ``x, x'``, the rational kernel with shape parameter
``\\alpha > 0`` is defined as
```math
k(x, x'; \\alpha) = \\bigg(1 + \\frac{\\|x - x'\\|}{\\alpha}\\bigg)^{-\\alpha}.
```

The [`ExponentialKernel`](@ref) is recovered in the limit as ``\\alpha \\to \\infty``.

See also: [`GammaRationalKernel`](@ref)
"""
struct RationalKernel{Tα<:Real,M} <: SimpleKernel
    α::Vector{Tα}
    metric::M

    function RationalKernel(; alpha::Real=2.0, α::Real=alpha, metric=Euclidean())
        @check_args(RationalKernel, α, α > zero(α), "α > 0")
        return new{typeof(α),typeof(metric)}([α], metric)
    end
end

@functor RationalKernel

function kappa(κ::RationalKernel, d::Real)
    return (one(d) + d / first(κ.α))^(-first(κ.α))
end

metric(k::RationalKernel) = k.metric

function Base.show(io::IO, κ::RationalKernel)
    return print(io, "Rational Kernel (α = ", first(κ.α), ", metric = ", κ.metric, ")")
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
    return (one(d) + d^2 / (2 * first(κ.α)))^(-first(κ.α))
end
function kappa(κ::RationalQuadraticKernel{<:Real,<:Euclidean}, d²::Real)
    return (one(d²) + d² / (2 * first(κ.α)))^(-first(κ.α))
end

metric(k::RationalQuadraticKernel) = k.metric
metric(::RationalQuadraticKernel{<:Real,<:Euclidean}) = SqEuclidean()

function Base.show(io::IO, κ::RationalQuadraticKernel)
    return print(
        io, "Rational Quadratic Kernel (α = ", first(κ.α), ", metric = ", κ.metric, ")"
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
    return (one(d) + d^first(κ.γ) / first(κ.α))^(-first(κ.α))
end

metric(k::GammaRationalKernel) = k.metric

function Base.show(io::IO, κ::GammaRationalKernel)
    return print(
        io,
        "Gamma Rational Kernel (α = ",
        first(κ.α),
        ", γ = ",
        first(κ.γ),
        ", metric = ",
        κ.metric,
        ")",
    )
end
