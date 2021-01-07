"""
    SqExponentialKernel()

The squared exponential kernel is a Mercer kernel given by the formula:
```
    κ(x, y) = exp(-‖x - y‖² / 2)
```
Can also be called via `RBFKernel`, `GaussianKernel` or `SEKernel`.
See [`GammaExponentialKernel`](@ref) for a generalization.
"""
struct SqExponentialKernel <: SimpleKernel end

kappa(κ::SqExponentialKernel, d²::Real) = exp(-d² / 2)

metric(::SqExponentialKernel) = SqEuclidean()

iskroncompatible(::SqExponentialKernel) = true

Base.show(io::IO,::SqExponentialKernel) = print(io,"Squared Exponential Kernel")

## Aliases ##

"""
    RBFKernel()

See [`SqExponentialKernel`](@ref)
"""
const RBFKernel = SqExponentialKernel

"""
    GaussianKernel()

See [`SqExponentialKernel`](@ref)
"""
const GaussianKernel = SqExponentialKernel

"""
    SEKernel()

See [`SqExponentialKernel`](@ref)
"""
const SEKernel = SqExponentialKernel


"""
    ExponentialKernel()

The exponential kernel is a Mercer kernel given by the formula:
```
    κ(x,y) = exp(-‖x-y‖)
```
Can also be called via `LaplacianKernel` or `Matern12Kernel`.
See [`GammaExponentialKernel`](@ref) for a generalization.
"""
struct ExponentialKernel <: SimpleKernel end

kappa(κ::ExponentialKernel, d::Real) = exp(-d)

metric(::ExponentialKernel) = Euclidean()

iskroncompatible(::ExponentialKernel) = true

Base.show(io::IO, ::ExponentialKernel) = print(io, "Exponential Kernel")

## Aliases ##

"""
    LaplacianKernel()

See [`ExponentialKernel`](@ref)
"""
const LaplacianKernel = ExponentialKernel

"""
    Matern12Kernel()

See [`ExponentialKernel`](@ref)
"""
const Matern12Kernel = ExponentialKernel


"""
    GammaExponentialKernel(; γ = 2.0)

The γ-exponential kernel [1] is an isotropic Mercer kernel given by the formula:
```
    κ(x,y) = exp(-‖x-y‖^γ)
```
Where `γ > 0`, (the keyword `γ` can be replaced by `gamma`)
For `γ = 2`, see [`SqExponentialKernel`](@ref); for `γ = 1`, see [`ExponentialKernel`](@ref).

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
