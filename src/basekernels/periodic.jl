"""
    PeriodicKernel(r::AbstractVector)
    PeriodicKernel(dims::Int)
    PeriodicKernel(T::DataType, dims::Int)

Periodic Kernel as described in http://www.inference.org.uk/mackay/gpB.pdf eq. 47.
```
    κ(x,y) = exp( - 0.5 sum_i(sin (π(x_i - y_i))/r_i))
```
"""
struct PeriodicKernel{T} <: BaseKernel
    r::Vector{T}
    function PeriodicKernel(; r::AbstractVector{T} = ones(Float64, 1)) where {T<:Real}
        @assert all(r .> 0)
        new{T}(r)
    end
end

PeriodicKernel(dims::Int) = PeriodicKernel(Float64, dims)

PeriodicKernel(T::DataType, dims::Int = 1) = PeriodicKernel(r = ones(T, dims))

metric(κ::PeriodicKernel) = Sinus(κ.r)

kappa(κ::PeriodicKernel, d::Real) = exp(- 0.5d)

Base.show(io::IO, κ::PeriodicKernel) = print(io, "Periodic Kernel, length(r) = ", length(κ.r), ")")
