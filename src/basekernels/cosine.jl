"""
    CosineKernel()

The cosine kernel is a stationary kernel for a sinusoidal given by
```
    κ(x,y) = cos( π * (x-y) )
```
"""
struct CosineKernel <: SimpleKernel end

kappa(κ::CosineKernel, d::Real) = cospi(d)

metric(::CosineKernel) = Euclidean()

(k::CosineKernel)(x, y) = eval_fallback(k, x, y)

Base.show(io::IO, ::CosineKernel) = print(io, "Cosine Kernel")
