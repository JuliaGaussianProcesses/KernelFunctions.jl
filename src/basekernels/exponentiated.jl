"""
    ExponentiatedKernel()

The exponentiated kernel is a Mercer kernel given by:
```
    κ(x,y) = exp(xᵀy)
```
"""
struct ExponentiatedKernel <: SimpleKernel end

kappa(κ::ExponentiatedKernel, xᵀy::Real) = exp(xᵀy)

metric(::ExponentiatedKernel) = DotProduct()

(k::ExponentiatedKernel)(x, y) = eval_fallback(k, x, y)

iskroncompatible(::ExponentiatedKernel) = true

Base.show(io::IO, ::ExponentiatedKernel) = print(io, "Exponentiated Kernel")
