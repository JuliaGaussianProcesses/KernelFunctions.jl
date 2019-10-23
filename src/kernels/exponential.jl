"""
    SqExponentialKernel([ρ=1.0])

The squared exponential kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = exp(-‖x-y‖²)
```

See also [`ExponentialKernel`](@ref) for a
related form of the kernel or [`GammaExponentialKernel`](@ref) for a generalization.
"""
struct SqExponentialKernel{T,Tr} <: Kernel{T,Tr}
    transform::Tr
    metric::SqEuclidean
    function SqExponentialKernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform,SqEuclidean())
    end
end

@inline kappa(κ::SqExponentialKernel, d²::Real) where {T} = exp(-d²)

### Aliases
const RBFKernel = SqExponentialKernel
const GaussianKernel = SqExponentialKernel

"""
    ExponentialKernel([ρ=1.0])

The exponential kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = exp(-‖x-y‖)
```

"""
struct ExponentialKernel{T,Tr} <: Kernel{T,Tr}
    transform::Tr
    metric::Euclidean
    function ExponentialKernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform,Euclidean())
    end
end

@inline kappa(κ::ExponentialKernel, d::Real) where {T} = exp(-d)

### Aliases
const LaplacianKernel = ExponentialKernel

"""
    GammaExponentialKernel([ρ=1.0,[gamma=2.0]])

The γ-exponential kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = exp(-‖x-y‖^2γ)
```
"""
struct GammaExponentialKernel{T,Tr,Tᵧ<:Real} <: Kernel{T,Tr}
    transform::Tr
    metric::SqEuclidean
    γ::Tᵧ
    function GammaExponentialKernel{T,Tr,Tᵧ}(transform::Tr,γ::Tᵧ) where {T,Tr<:Transform,Tᵧ<:Real}
        return new{T,Tr,Tᵧ}(transform,SqEuclidean(),γ)
    end
end

function GammaExponentialKernel(ρ::T₁=1.0,gamma::T₂=2.0) where {T₁<:Real,T₂<:Real}
    @check_args(GammaExponentialKernel, gamma, gamma >= zero(T₂), "gamma > 0")
    GammaExponentialKernel{T₁,ScaleTransform{T₁},T₂}(ScaleTransform(ρ),gamma)
end

function GammaExponentialKernel(ρ::A,gamma::T₁=2.0) where {A<:AbstractVector{<:Real},T₁<:Real}
    @check_args(GammaExponentialKernel, gamma, gamma >= zero(T₁), "gamma > 0")
    GammaExponentialKernel{eltype(A),ScaleTransform{A},T₁}(ScaleTransform(ρ),gamma)
end

function GammaExponentialKernel(t::Tr,gamma::T₁=2.0) where {Tr<:Transform,T₁<:Real}
    @check_args(GammaExponentialKernel, gamma, gamma >= zero(T₁), "gamma > 0")
    GammaExponentialKernel{eltype(Tr),Tr,T₁}(t,gamma)
end

@inline kappa(κ::GammaExponentialKernel, d²::Real) where {T} = exp(-d²^κ.γ)
