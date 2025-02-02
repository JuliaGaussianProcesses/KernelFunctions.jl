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

PeriodicKernel(dims::Int) = PeriodicKernel(Float64, dims)

"""
    PeriodicKernel([T=Float64, dims::Int=1])

Create a [`PeriodicKernel`](@ref) with parameter `r=ones(T, dims)`.
"""
PeriodicKernel(T::DataType, dims::Int=1) = PeriodicKernel(; r=ones(T, dims))

@functor PeriodicKernel

metric(κ::PeriodicKernel) = Sinus(κ.r)

kappa(::PeriodicKernel, d::Real) = exp(-d / 2)

function Base.show(io::IO, κ::PeriodicKernel)
    return print(io, "Periodic Kernel, length(r) = $(length(κ.r))")
end
