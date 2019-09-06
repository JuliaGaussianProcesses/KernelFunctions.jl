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
    metric::SemiMetric
    function SqExponentialKernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform,SqEuclidean())
    end
end

function SqExponentialKernel(α::T=1.0) where {T<:Real}
    SqExponentialKernel{T,ScaleTransform{T}}(ScaleTransform(α))
end

function SqExponentialKernel(α::A) where {A<:AbstractVector{<:Real}}
    SqExponentialKernel{eltype(A),ScaleTransform{A}}(ScaleTransform(α))
end

function SqExponentialKernel(t::T) where {T<:Transform}
    SqExponentialKernel{eltype(t),T}(t)
end

@inline kappa(κ::SqExponentialKernel, d²::Real) where {T} = exp(-d²)

# function convert(
#         ::Type{K},
#         κ::SqExponentialKernel
#     ) where {K>:SqExponentialKernel{T,A} where {T,A}}
#     return SqExponentialKernel{T}(T.(κ.α))
# end
