"""
    CosineKernel()

The cosine kernel is a stationary kernel for a sinusoidal given by
```
    κ(x,y) = cos( π * (x-y) )
```
"""
struct CosineKernel <: BaseKernel end

kappa(κ::CosineKernel, d::Real) = cospi(d)
metric(::CosineKernel) = Euclidean()

Base.show(io::IO, ::CosineKernel) = print(io, "Cosine Kernel")
