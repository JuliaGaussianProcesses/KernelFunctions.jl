"""
    MaternKernel([[ρ=1],ν=3/2])

The matern kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = 2^{1-ν}/Γ(ν)*(√(2ν)‖x-y‖/ρ)^ν K_ν(√(2ν)‖x-y‖/ρ)
```

For `ν=n+1/2, n=0,1,2,...` it can be simplified (it will be converted automatically).
`ρ` is a lengthscale parameter.

# Examples

```jldoctest; setup = :(using KernelFunctions)
julia> MaternKernel()
Matern3_2Kernel{Float64,Float64}(1.0)

julia> MaternKernel(2.0f0,3.0)
MaternKernel{Float32,Float32}(2.0,3.0)

julia> MaternKernel([2.0,3.0],5/2)
Matern5_2Kernel{Float64,Array{Float64}}([2.0,3.0])
```
"""
struct MaternKernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::SemiMetric
    ν::Real
    function MaternKernel{T,Tr}(transform::Tr,ν::Real) where {T,Tr<:Transform}
        return new{T,Tr}(transform,Euclidean(),ν)
    end
end

function MaternKernel(ρ::T₁=1.0,ν::T₂=1.5) where {T₁<:Real,T₂<:Real}
    @check_args(MaternKernel, ν, ν > zero(T₂), "ν > 0")
    if ν == 0.5
        ExponentialKernel{T₁,ScaleTransform{T₁}}(ScaleTransform(ρ))
    elseif ν == 1.5
        Matern3_2Kernel{T₁,ScaleTransform{T₁}}(ScaleTransform(ρ))
    elseif ν == 2.5
        Matern5_2Kernel{T₁,ScaleTransform{T₁}}(ScaleTransform(ρ))
    else
        MaternKernel{T₁,ScaleTransform{T₁}}(ScaleTransform(ρ),ν)
    end
end

function MaternKernel(ρ::A,ν::T=1.5) where {A<:AbstractVector{<:Real},T<:Real}
    @check_args(MaternKernel, ν, ν > zero(T), "ν > 0")
    if ν == 0.5
        ExponentialKernel{eltype(A),ScaleTransform{A}}(ScaleTransform(ρ))
    elseif ν == 1.5
        Matern3_2Kernel{eltype(A),ScaleTransform{A}}(ScaleTransform(ρ))
    elseif ν == 2.5
        Matern5_2Kernel{eltype(A),ScaleTransform{A}}(ScaleTransform(ρ))
    else
        MaternKernel{eltype(A),ScaleTransform{A}}(ScaleTransform(ρ),ν)
    end
end

function MaternKernel(t::T₁,ν::T₂=1.5) where {T₁<:Transform,T₂<:Real}
    @check_args(MaternKernel, ν, ν > zero(T₂), "ν > 0")
    if ν == 0.5
        ExponentialKernel{eltype(t),T₁}(ScaleTransform(ρ))
    elseif ν == 1.5
        Matern3_2Kernel{eltype(t),T₁}(ScaleTransform(ρ))
    elseif ν == 2.5
        Matern5_2Kernel{eltype(t),T₁}(ScaleTransform(ρ))
    else
        MaternKernel{eltype(t),T₁}(ScaleTransform(ρ),ν)
    end
end

@inline kappa(κ::MaternKernel, d::Real) where {T} = exp((1.0-κ.ν)*logtwo - lgamma(κ.ν) - κ.ν*log(sqrt(2κ.ν)*d))*besselk(κ.ν,sqrt(2κ.ν)*d)


struct Matern3_2Kernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::SemiMetric
    function Matern3_2Kernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform,Euclidean())
    end
end

@inline kappa(κ::Matern3_2Kernel, d::T) where {T<:Real} = (1+sqrt(3)*d)*exp(-sqrt(3)*d)

struct Matern5_2Kernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::SemiMetric
    function Matern5_2Kernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform,Euclidean())
    end
end

@inline kappa(κ::Matern5_2Kernel, d::Real) where {T} = (1+sqrt(5)*d+5*d^2/3)*exp(-sqrt(5)*d)
