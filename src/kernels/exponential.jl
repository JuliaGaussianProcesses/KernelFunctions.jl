"""
`SqExponentialKernel()`

The squared exponential kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y) = exp(-‖x-y‖²)
```
See also [`ExponentialKernel`](@ref) for a
related form of the kernel or [`GammaExponentialKernel`](@ref) for a generalization.
"""
struct SqExponentialKernel <: Kernel end

@inline kappa(κ::SqExponentialKernel, d²::Real) = exp(-d²)
@inline iskroncompatible(::SqExponentialKernel) = true

metric(::SqExponentialKernel) = SqEuclidean()

## Aliases ##
const RBFKernel = SqExponentialKernel
const GaussianKernel = SqExponentialKernel

"""
`ExponentialKernel([ρ=1.0])`
The exponential kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y) = exp(-ρ‖x-y‖)
```
"""
struct ExponentialKernel <: Kernel end

@inline kappa(κ::ExponentialKernel, d::Real) = exp(-d)
@inline iskroncompatible(::ExponentialKernel) = true
metric(::ExponentialKernel) = Euclidean()

## Alias ##
const LaplacianKernel = ExponentialKernel

"""
`GammaExponentialKernel([ρ=1.0, [γ=2.0]])`
The γ-exponential kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y) = exp(-ρ^(2γ)‖x-y‖^(2γ))
```
"""
struct GammaExponentialKernel{Tγ<:Real} <: Kernel
    γ::Tγ
    function GammaExponentialKernel(γ::T=2.0) where {T<:Real}
        @check_args(GammaExponentialKernel, γ, γ >= zero(T), "γ > 0")
        return new{T}(γ)
    end
end

params(k::GammaExponentialKernel) = (γ)
opt_params(k::GammaExponentialKernel) = (γ)

@inline kappa(κ::GammaExponentialKernel, d²::Real) = exp(-d²^κ.γ)
@inline iskroncompatible(::GammaExponentialKernel) = true
metric(::GammaExponentialKernel) = SqEuclidean()
