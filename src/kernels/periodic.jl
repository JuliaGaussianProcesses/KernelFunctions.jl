"""
    PeriodicKernel(r::AbstractVector)
    PeriodicKernel(dims::Int)
```
    κ(x,y) = exp(-2 * sum_i(sin (π(x_i - y_i))/r_i))
```
"""
struct PeriodicKernel{T} <: BaseKernel
    r::Vector{T}
    function PeriodicKernel(; r::AbstractVector{T} = ones(Float64, 1)) where {T<:Real}
        @assert all(r .> 0)
        new{T}(r)
    end
end

PeriodicKernel(dims::Int = 1) = PeriodicKernel(Float64, dims)

PeriodicKernel(T::DataType, dims::Int = 1) = PeriodicKernel(r = ones(T, dims))

metric(κ::PeriodicKernel) = Sinus(κ.r)

kappa(κ::PeriodicKernel, d::Real) = exp(-0.5 * d)
