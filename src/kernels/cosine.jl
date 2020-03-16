"""
    CosineKernel

The cosine kernel is a stationary kernel for a sinusoidal with period p in 1d given by
```
    κ(x,y) = sf² * cos( 2π * (x-y) / p )
```

Where `p` is the period and `sf` is the scaling factor.
"""
struct CosineKernel <: BaseKernel end

kappa(κ::CosineKernel, d::Real) = cos(2*pi*d)

metric(::CosineKernel) = Cityblock()
