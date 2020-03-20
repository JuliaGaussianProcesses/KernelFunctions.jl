"""
    CosineKernel

The cosine kernel is a stationary kernel for a sinusoidal given by
```
    κ(x,y) = cos( 2π * (x-y) )
```

"""
struct CosineKernel <: BaseKernel end

kappa(κ::CosineKernel, d::Real) = cos(2*pi*d)

metric(::CosineKernel) = Euclidean()
