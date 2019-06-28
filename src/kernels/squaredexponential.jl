"""
    SquaredExponentialKernel([α=1])

The squared exponential kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = exp(-‖x-y‖²)
```

where `α` is a positive scaling parameter. See also [`ExponentialKernel`](@ref) for a
related form of the kernel or [`GammaExponentialKernel`](@ref) for a generalization.

# Examples

```jldoctest; setup = :(using KernelFunctions)
julia> SquaredExponentialKernel()
SquaredExponentialKernel{Float64,Float64}(1.0)

julia> SquaredExponentialKernel(2.0f0)
SquaredExponentialKernel{Float32,Float32}(2.0)

julia> SquaredExponentialKernel([2.0,3.0])
SquaredExponentialKernel{Float64,Array{Float64}}(1.0)
```
"""
struct SquaredExponentialKernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::SemiMetric
    function SquaredExponentialKernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform,SqEuclidean())
    end
end

function SquaredExponentialKernel(α::T=1.0) where {T<:Real}
    SquaredExponentialKernel{T,ScaleTransform{T}}(ScaleTransform(α))
end

function SquaredExponentialKernel(α::A) where {A<:AbstractVector{<:Real}}
    SquaredExponentialKernel{eltype(A),ScaleTransform{A}}(ScaleTransform(α))
end

function SquaredExponentialKernel(t::T) where {T<:Transform}
    SquaredExponentialKernel{eltype(t),T}(t)
end

@inline kappa(κ::SquaredExponentialKernel, d²::Real) where {T} = exp(-d²)

# function convert(
#         ::Type{K},
#         κ::SquaredExponentialKernel
#     ) where {K>:SquaredExponentialKernel{T,A} where {T,A}}
#     return SquaredExponentialKernel{T}(T.(κ.α))
# end
