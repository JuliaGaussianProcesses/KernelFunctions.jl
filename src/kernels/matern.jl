"""
`MaternKernel([ρ=1.0,[ν=1.0]])`
The matern kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y) = 2^{1-ν}/Γ(ν)*(√(2ν)‖x-y‖)^ν K_ν(√(2ν)‖x-y‖)
```
For `ν=n+1/2, n=0,1,2,...` it can be simplified and you should instead use [`ExponentialKernel`](@ref) for `n=0`, [`Matern32Kernel`](@ref), for `n=1`, [`Matern52Kernel`](@ref) for `n=2` and [`SqExponentialKernel`](@ref) for `n=∞`.
"""
struct MaternKernel{T,Tr,Tν<:Real} <: Kernel{T,Tr}
    transform::Tr
    metric::Euclidean
    ν::Tν
    function MaternKernel{T,Tr,Tν}(transform::Tr,ν::Tν) where {T,Tr<:Transform,Tν<:Real}
        return new{T,Tr,Tν}(transform,Euclidean(),ν)
    end
end

function MaternKernel(ρ::T₁=1.0,ν::T₂=1.5) where {T₁<:Real,T₂<:Real}
    @check_args(MaternKernel, ν, ν > zero(T₂), "ν > 0")
    MaternKernel{T₁,ScaleTransform{Base.RefValue{T₁}},T₂}(ScaleTransform(ρ),ν)
end

function MaternKernel(ρ::A,ν::T=1.5) where {A<:AbstractVector{<:Real},T<:Real}
    @check_args(MaternKernel, ν, ν > zero(T), "ν > 0")
    MaternKernel{eltype(A),ScaleTransform{A},T}(ScaleTransform(ρ),ν)
end

function MaternKernel(t::Tr,ν::T=1.5) where {Tr<:Transform,T<:Real}
    @check_args(MaternKernel, ν, ν > zero(T), "ν > 0")
    MaternKernel{eltype(t),Tr,T}(t,ν)
end

@inline kappa(κ::MaternKernel, d::Real) = iszero(d) ? one(d) : exp((1.0-κ.ν)*logtwo-lgamma(κ.ν) + κ.ν*log(sqrt(2κ.ν)*d)+log(besselk(κ.ν,sqrt(2κ.ν)*d)))

"""
`Matern32Kernel([ρ=1.0])`
The matern 3/2 kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y) = (1+√(3)ρ‖x-y‖)exp(-√(3)ρ‖x-y‖)
```
"""
struct Matern32Kernel{T,Tr} <: Kernel{T,Tr}
    transform::Tr
    metric::Euclidean
    function Matern32Kernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform,Euclidean())
    end
end

@inline kappa(κ::Matern32Kernel, d::T) where {T<:Real} = (1+sqrt(3)*d)*exp(-sqrt(3)*d)

"""
`Matern52Kernel([ρ=1.0])`
The matern 5/2 kernel is an isotropic Mercer kernel given by the formula:
```
    κ(x,y) = (1+√(5)ρ‖x-y‖ + 5ρ²‖x-y‖^2/3)exp(-√(5)ρ‖x-y‖)
```
"""
struct Matern52Kernel{T,Tr} <: Kernel{T,Tr}
    transform::Tr
    metric::Euclidean
    function Matern52Kernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform,Euclidean())
    end
end

@inline kappa(κ::Matern52Kernel, d::Real) where {T} = (1+sqrt(5)*d+5*d^2/3)*exp(-sqrt(5)*d)
