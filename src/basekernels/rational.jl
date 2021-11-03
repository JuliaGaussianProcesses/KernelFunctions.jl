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
struct RationalKernel{T<:Real,M} <: SimpleKernel
    α::T
    metric::M

    function RationalKernel(α::Real, metric)
        @check_args(RationalKernel, α, α > zero(α), "α > 0")
        return new{typeof(α),typeof(metric)}(α, metric)
    end
end

function RationalKernel(; alpha::Real=2.0, α::Real=alpha, metric=Euclidean())
    return RationalKernel(α, metric)
end

function ParameterHandling.flatten(::Type{T}, k::RationalKernel{S}) where {T<:Real,S}
    metric = k.metric
    function unflatten_to_rationalkernel(v::Vector{T})
        length(v) == 1 || error("incorrect number of parameters")
        return ConstantKernel(S(exp(first(v))), metric)
    end
    return T[log(k.α)], unflatten_to_rationalkernel
end

function kappa(κ::RationalKernel, d::Real)
    α = κ.α
    return (one(d) + d / α)^(-α)
end

metric(k::RationalKernel) = k.metric

function Base.show(io::IO, κ::RationalKernel)
    return print(io, "Rational Kernel (α = ", κ.α, ", metric = ", κ.metric, ")")
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
struct RationalQuadraticKernel{T<:Real,M} <: SimpleKernel
    α::T
    metric::M

    function RationalQuadraticKernel(; alpha::Real=2.0, α::Real=alpha, metric=Euclidean())
        @check_args(RationalQuadraticKernel, α, α > zero(α), "α > 0")
        return new{typeof(α),typeof(metric)}(α, metric)
    end
end

function ParameterHandling.flatten(
    ::Type{T}, k::RationalQuadraticKernel{S}
) where {T<:Real,S}
    metric = k.metric
    function unflatten_to_rationalquadratickernel(v::Vector{T})
        length(v) == 1 || error("incorrect number of parameters")
        return RationalQuadraticKernel(; α=S(exp(first(v))), metric=metric)
    end
    return T[log(k.α)], unflatten_to_rationalquadratickernel
end

function kappa(κ::RationalQuadraticKernel, d::Real)
    α = κ.α
    return (one(d) + d^2 / (2 * α))^(-α)
end
function kappa(κ::RationalQuadraticKernel{<:Real,<:Euclidean}, d²::Real)
    α = κ.α
    return (one(d²) + d² / (2 * α))^(-α)
end

metric(k::RationalQuadraticKernel) = k.metric
metric(::RationalQuadraticKernel{<:Real,<:Euclidean}) = SqEuclidean()

function Base.show(io::IO, κ::RationalQuadraticKernel)
    return print(io, "Rational Quadratic Kernel (α = ", κ.α, ", metric = ", κ.metric, ")")
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
    α::Tα
    γ::Tγ
    metric::M

    function GammaRationalKernel(;
        alpha::Real=2.0, gamma::Real=1.0, α::Real=alpha, γ::Real=gamma, metric=Euclidean()
    )
        @check_args(GammaRationalKernel, α, α > zero(α), "α > 0")
        @check_args(GammaRationalKernel, γ, zero(γ) < γ ≤ 2, "γ ∈ (0, 2]")
        return new{typeof(α),typeof(γ),typeof(metric)}(α, γ, metric)
    end
end

@functor GammaRationalKernel

function ParameterHandling.flatten(
    ::Type{T}, k::GammaRationalKernel{Tα,Tγ}
) where {T<:Real,Tα,Tγ}
    vec = T[log(k.α), logit(k.γ - 1)]
    metric = k.metric
    function unflatten_to_gammarationalkernel(v::Vector{T})
        length(v) == 2 || error("incorrect number of parameters")
        logα, logitγ = v
        α = Tα(exp(logα))
        γ = Tγ(1 + logistic(logitγ))
        return GammaRationalKernel(; α=α, γ=γ, metric=metric)
    end
    return vec, unflatten_to_gammarationalkernel
end

function kappa(κ::GammaRationalKernel, d::Real)
    α = κ.α
    return (one(d) + d^κ.γ / α)^(-α)
end

metric(k::GammaRationalKernel) = k.metric

function Base.show(io::IO, κ::GammaRationalKernel)
    return print(
        io, "Gamma Rational Kernel (α = ", κ.α, ", γ = ", κ.γ, ", metric = ", κ.metric, ")"
    )
end
