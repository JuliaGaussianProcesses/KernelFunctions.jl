"""
`SqExponentialKernel([ρ=1.0])`

The squared exponential kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y) = exp(-ρ²‖x-y‖²)
```
See also [`ExponentialKernel`](@ref) for a
related form of the kernel or [`GammaExponentialKernel`](@ref) for a generalization.
"""
struct SqExponentialKernel{Tr} <: Kernel{Tr}
    transform::Tr
end

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
struct ExponentialKernel{Tr} <: Kernel{Tr}
    transform::Tr
end

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
struct GammaExponentialKernel{Tr, Tγ<:Real} <: Kernel{Tr}
    transform::Tr
    γ::Tγ
    function GammaExponentialKernel{Tr,Tγ}(t::Tr, γ::Tγ) where {Tr<:Transform,Tγ<:Real}
        @check_args(GammaExponentialKernel, γ, γ >= zero(Tγ), "γ > 0")
        return new{Tr, Tγ}(t, γ)
    end
end

params(k::GammaExponentialKernel) = (params(transform),γ)
opt_params(k::GammaExponentialKernel) = (opt_params(transform),γ)

function GammaExponentialKernel(ρ::Real=1.0, γ::Real=2.0)
    GammaExponentialKernel(ScaleTransform(ρ), γ)
end

function GammaExponentialKernel(ρ::AbstractVector{<:Real}, γ::Real=2.0)
    GammaExponentialKernel(ARDTransform(ρ), γ)
end

function GammaExponentialKernel(t::Tr, γ::Tγ=2.0) where {Tr<:Transform, Tγ<:Real}
    GammaExponentialKernel{Tr, Tγ}(t, γ)
end

@inline kappa(κ::GammaExponentialKernel, d²::Real) = exp(-d²^κ.γ)
@inline iskroncompatible(::GammaExponentialKernel) = true
metric(::GammaExponentialKernel) = SqEuclidean()
