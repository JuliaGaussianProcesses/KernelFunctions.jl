"""
`SqExponentialKernel()`

The squared exponential kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y) = exp(-‖x-y‖²)
```
See also [`ExponentialKernel`](@ref) for a
related form of the kernel or [`GammaExponentialKernel`](@ref) for a generalization.
"""
struct SqExponentialKernel <: BaseKernel end

kappa(κ::SqExponentialKernel, d²::Real) = exp(-d²)
iskroncompatible(::SqExponentialKernel) = true

metric(::SqExponentialKernel) = SqEuclidean()

Base.show(io::IO,::SqExponentialKernel) = print(io,"Squared Exponential Kernel")

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
struct ExponentialKernel <: BaseKernel end

kappa(κ::ExponentialKernel, d::Real) = exp(-d)
iskroncompatible(::ExponentialKernel) = true
metric(::ExponentialKernel) = Euclidean()

Base.show(io::IO,::ExponentialKernel) = print(io,"Exponential Kernel")

## Alias ##
const LaplacianKernel = ExponentialKernel

"""
`GammaExponentialKernel([ρ=1.0, [γ=2.0]])`
The γ-exponential kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y) = exp(-ρ^(2γ)‖x-y‖^(2γ))
```
"""
struct GammaExponentialKernel{Tγ<:Real} <: BaseKernel
    γ::Tγ
    function GammaExponentialKernel(;γ::T=2.0) where {T<:Real}
        @check_args(GammaExponentialKernel, γ, γ >= zero(T), "γ > 0")
        return new{T}(γ)
    end
end

params(k::GammaExponentialKernel) = (γ,)
opt_params(k::GammaExponentialKernel) = (γ,)

kappa(κ::GammaExponentialKernel, d²::Real) = exp(-d²^κ.γ)
iskroncompatible(::GammaExponentialKernel) = true
metric(::GammaExponentialKernel) = SqEuclidean()
