@doc raw"""
    SquaredExponentialKernel([α=1])

The squared exponential kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = exp(α‖x-y‖²)   α > 0
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
struct SquaredExponentialKernel{T<:Real,A} <: Kernel{T}
    α::A
    metric::SemiMetric
    function SquaredExponentialKernel{T}(α::A=T(1)) where {A<:Union{Real,AbstractVector{<:Real}},T<:Real}
        @check_args(SquaredExponentialKernel, α, all(α .> zero(T)), "α > 0")
        if A <: Real
            return new{eltype(A),A}(α,SqEuclidean())
        else
            return new{eltype(A),A}(α,WeightedSqEuclidean(α))
        end
    end
end

function SquaredExponentialKernel(α::Union{T,AbstractVector{T}}=1.0) where {T<:Real}
    SquaredExponentialKernel{promote_float(T)}(α)
end

@inline kappa(κ::SquaredExponentialKernel{T,<:Real}, d²::Real) where {T} = exp(-κ.α*d²)
@inline kappa(κ::SquaredExponentialKernel{T}, d²::Real) where {T} = exp(-d²)

function convert(
        ::Type{K},
        κ::SquaredExponentialKernel
    ) where {K>:SquaredExponentialKernel{T,A} where {T,A}}
    return SquaredExponentialKernel{T}(T.(κ.α))
end
