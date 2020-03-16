"""
`CosineKernel()`
The cosine kernel is a stationary kernel for a sinusoidal with period p in 1d given by
```
    κ(x,y) = sf² * cos( 2π * (x-y) / p )
```

Where `p` is the period and `sf` is the scaling factor.
"""
struct CosineKernel <: BaseKernel
    p::Real
    sf::Real
end

kappa(κ::CosineKernel, d::Real) = κ.sf*κ.sf*cos(2*pi*d/κ.p)

metric(::CosineKernel) = Cityblock()
