"""
    CosineKernel()

The cosine kernel is a stationary kernel for a sinusoidal given by
```
    κ(x,y) = cos( π * (x-y) )
```
"""
struct CosineKernel <: SimpleKernel end

kappa(κ::CosineKernel, d::Real) = cospi(d)

binary_op(::CosineKernel) = Euclidean()

Base.show(io::IO, ::CosineKernel) = print(io, "Cosine Kernel")
