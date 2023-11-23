"""
    PeriodicKernel(; r::AbstractVector=ones(Float64, 1))

Periodic kernel with parameter `r`.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the periodic kernel with parameter ``r_i > 0`` is
defined[^DM] as
```math
k(x, x'; r) = \\exp\\bigg(- \\frac{1}{2} \\sum_{i=1}^d \\bigg(\\frac{\\sin\\big(\\pi(x_i - x'_i)\\big)}{r_i}\\bigg)^2\\bigg).
```

[^DM]: D. J. C. MacKay (1998). Introduction to Gaussian Processes.
"""
struct PeriodicKernel{T} <: SimpleKernel
    r::Vector{T}
    function PeriodicKernel(; r::AbstractVector{<:Real}=ones(Float64, 1))
        @check_args(PeriodicKernel, r, all(ri > zero(ri) for ri in r), "r > 0")
        return new{eltype(r)}(r)
    end
end

"""
    PeriodicKernel(dims::Int)

Create a [`PeriodicKernel`](@ref) with parameter `r=ones(Float64, dims)`.
"""
PeriodicKernel(dims::Int) = PeriodicKernel(Float64, dims)

"""
    PeriodicKernel(T, dims::Int=1)

Create a [`PeriodicKernel`](@ref) with parameter `r=ones(T, dims)`.
"""
PeriodicKernel(::Type{T}, dims::Int=1) where {T} = PeriodicKernel(; r=ones(T, dims))

function ParameterHandling.flatten(::Type{T}, k::PeriodicKernel{S}) where {T<:Real,S}
    vec = T[log(ri) for ri in k.r]
    unflatten_to_periodickernel(v::Vector{T}) = PeriodicKernel(; r=S[exp(vi) for vi in v])
    return vec, unflatten_to_periodickernel
end

metric(κ::PeriodicKernel) = Sinus(κ.r)

kappa(::PeriodicKernel, d::Real) = exp(-0.5d)

function Base.show(io::IO, κ::PeriodicKernel)
    return print(io, "Periodic Kernel, length(r) = $(length(κ.r))")
end
