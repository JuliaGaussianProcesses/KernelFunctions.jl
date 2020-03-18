"""
    PeriodicKernel(r::AbstractVector)
    PeriodicKernel(dims::Int)
```
    κ(x,y) = exp(-2 * sum_i(sin (π(x_i - y_i))/r_i))
```
"""
struct PeriodicKernel{T} <: BaseKernel
    r::Vector{T}
end

PeriodicKernel(dims::Int) = PeriodicKernel{Float64}(ones(Float64,dims))

metric(κ::PeriodicKernel) = Sinus(κ.r)

kappa(κ::PeriodicKernel, d::Real) = exp(-0.5 * d)
