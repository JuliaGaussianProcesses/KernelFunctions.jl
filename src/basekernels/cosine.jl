"""
    CosineKernel(; metric=Euclidean())

Cosine kernel with respect to the `metric`.

# Definition

For inputs ``x, x'``, the cosine kernel is defined as
```math
k(x, x') = \\cos(\\pi \\|x-x'\\|).
```
"""
struct CosineKernel{M} <: SimpleKernel
    metric::M

    function CosineKernel(; metric=Euclidean())
        return new{typeof(metric)}(metric)
    end
end

kappa(::CosineKernel, d::Real) = cospi(d)

metric(k::CosineKernel) = k.metric

Base.show(io::IO, k::CosineKernel) = print(io, "Cosine Kernel (metric = ", k.metric, ")")
