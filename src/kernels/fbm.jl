"""
    FBMKernel
Fractional Brownian motion kernel with Hurst index h from (0,1) given by
```
    κ(x,y) =  ( |x|²ʰ + |z|²ʰ - |x-z|²ʰ ) / 2
```

For h=1/2, this is the Wiener Kernel, for h>1/2, the increments are
positively correlated and for h<1/2 the increments are negatively correlated.
%
"""
struct FBMKernel{T<:Real} <: BaseKernel
    h::T
    function FBMKernel(;h::T=0.5) where {T<:Real}
        @assert h<=1.0 && h>=0.0, "FBMKernel: Given Hurst index h is invalid."
        new{T}(h)
    end
end

kappa(κ::FBMKernel, d::Real) = 0 # to be implemented

metric(::FBMKernel) = Euclidean()