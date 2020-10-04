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

kappa(κ::SqExponentialKernel, d²::Real) = exp(-d² / 2)

metric(::SqExponentialKernel) = SqEuclidean()

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

iskroncompatible(::ExponentialKernel) = true

Base.show(io::IO, ::ExponentialKernel) = print(io, "Exponential Kernel")

## Alias ##
const LaplacianKernel = ExponentialKernel

"""
    GammaExponentialKernel(; γ = 2.0)

The γ-exponential kernel [1] is an isotropic Mercer kernel given by the formula:
```
    κ(x,y) = exp(-‖x-y‖^γ)
```
Where `γ > 0`, (the keyword `γ` can be replaced by `gamma`)
For `γ = 2`, see `SqExponentialKernel` and `γ = 1`, see `ExponentialKernel`.

[1] - Gaussian Processes for Machine Learning, Carl Edward Rasmussen and Christopher K. I.
    Williams, MIT Press, 2006.
"""
struct GammaExponentialKernel{Tγ<:Real} <: SimpleKernel
    γ::Vector{Tγ}
    function GammaExponentialKernel(; gamma::T=2.0, γ::T=gamma) where {T<:Real}
        @check_args(GammaExponentialKernel, γ, γ >= zero(T), "γ > 0")
        return new{T}([γ])
    end
end

@functor GammaExponentialKernel

kappa(κ::GammaExponentialKernel, d::Real) = exp(-d^first(κ.γ))

metric(::GammaExponentialKernel) = Euclidean()

iskroncompatible(::GammaExponentialKernel) = true

function Base.show(io::IO, κ::GammaExponentialKernel)
    print(io, "Gamma Exponential Kernel (γ = ", first(κ.γ), ")")
end
