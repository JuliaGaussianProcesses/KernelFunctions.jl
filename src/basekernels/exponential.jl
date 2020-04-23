"""
    SqExponentialKernel()

The squared exponential kernel is a Mercer kernel given by the formula:
```
    κ(x,y) = exp(-‖x-y‖²)
```
Can also be called via `SEKernel`, `GaussianKernel` or `SEKernel`.
See also [`ExponentialKernel`](@ref) for a
related form of the kernel or [`GammaExponentialKernel`](@ref) for a generalization.
"""
struct SqExponentialKernel <: SimpleKernel end

kappa(κ::SqExponentialKernel, d²::Real) = exp(-d²)

metric(::SqExponentialKernel) = SqEuclidean()

(k::SqExponentialKernel)(x, y) = eval_fallback(k, x, y)

iskroncompatible(::SqExponentialKernel) = true

Base.show(io::IO,::SqExponentialKernel) = print(io,"Squared Exponential Kernel")

## Aliases ##
const RBFKernel = SqExponentialKernel
const GaussianKernel = SqExponentialKernel
const SEKernel = SqExponentialKernel

"""
    ExponentialKernel()

The exponential kernel is a Mercer kernel given by the formula:
```
    κ(x,y) = exp(-‖x-y‖)
```
"""
struct ExponentialKernel <: SimpleKernel end

kappa(κ::ExponentialKernel, d::Real) = exp(-d)

metric(::ExponentialKernel) = Euclidean()

(k::ExponentialKernel)(x, y) = eval_fallback(k, x, y)

iskroncompatible(::ExponentialKernel) = true

Base.show(io::IO, ::ExponentialKernel) = print(io, "Exponential Kernel")

## Alias ##
const LaplacianKernel = ExponentialKernel

"""
    GammaExponentialKernel(; γ = 2.0)

The γ-exponential kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y) = exp(-‖x-y‖^(2γ))
```
Where `γ > 0`, (the keyword `γ` can be replaced by `gamma`)
For `γ = 1`, see `SqExponentialKernel` and `γ = 0.5`, see `ExponentialKernel`
"""
struct GammaExponentialKernel{Tγ<:Real} <: SimpleKernel
    γ::Vector{Tγ}
    function GammaExponentialKernel(; gamma::T=2.0, γ::T=gamma) where {T<:Real}
        @check_args(GammaExponentialKernel, γ, γ >= zero(T), "γ > 0")
        return new{T}([γ])
    end
end

kappa(κ::GammaExponentialKernel, d²::Real) = exp(-d²^first(κ.γ))

metric(::GammaExponentialKernel) = SqEuclidean()

(k::GammaExponentialKernel)(x, y) = eval_fallback(k, x, y)

iskroncompatible(::GammaExponentialKernel) = true

Base.show(io::IO, κ::GammaExponentialKernel) = print(io, "Gamma Exponential Kernel (γ = ", first(κ.γ), ")")
