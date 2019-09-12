"""
    SqExponentialKernel([α=1])

The squared exponential kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = exp(-‖x-y‖²)
```

See also [`ExponentialKernel`](@ref) for a
related form of the kernel or [`GammaExponentialKernel`](@ref) for a generalization.

# Examples

```jldoctest; setup = :(using KernelFunctions)
julia> SqExponentialKernel()
SqExponentialKernel{Float64,Float64}(1.0)

julia> SqExponentialKernel(2.0f0)
SqExponentialKernel{Float32,Float32}(2.0)

julia> SqExponentialKernel([2.0,3.0])
SqExponentialKernel{Float64,Array{Float64}}(1.0)
```
"""
struct SqExponentialKernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::SqEuclidean
    function SqExponentialKernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform,SqEuclidean())
    end
end

function SqExponentialKernel(ρ::T=1.0) where {T<:Real}
    SqExponentialKernel{T,ScaleTransform{T}}(ScaleTransform(ρ))
end

function SqExponentialKernel(ρ::A) where {A<:AbstractVector{<:Real}}
    SqExponentialKernel{eltype(A),ScaleTransform{A}}(ScaleTransform(ρ))
end

function SqExponentialKernel(t::T) where {T<:Transform}
    SqExponentialKernel{eltype(t),T}(t)
end

@inline kappa(κ::SqExponentialKernel, d²::Real) where {T} = exp(-d²)

"""
    ExponentialKernel([α=1])

The exponential kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = exp(-‖x-y‖)
```

# Examples

```jldoctest; setup = :(using KernelFunctions)
julia> ExponentialKernel()
ExponentialKernel{Float64,Float64}(1.0)

julia> ExponentialKernel(2.0f0)
ExponentialKernel{Float32,Float32}(2.0)

julia> ExponentialKernel([2.0,3.0])
ExponentialKernel{Float64,Array{Float64}}(1.0)
```
"""
struct ExponentialKernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::Euclidean
    function ExponentialKernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform,Euclidean())
    end
end

function ExponentialKernel(α::T=1.0) where {T<:Real}
    ExponentialKernel{T,ScaleTransform{T}}(ScaleTransform(α))
end

function ExponentialKernel(α::A) where {A<:AbstractVector{<:Real}}
    ExponentialKernel{eltype(A),ScaleTransform{A}}(ScaleTransform(α))
end

function ExponentialKernel(t::T) where {T<:Transform}
    ExponentialKernel{eltype(t),T}(t)
end

@inline kappa(κ::ExponentialKernel, d::Real) where {T} = exp(-d)
