"""
    ExponentiatedKernel()

The exponentiated kernel is a Mercer kernel given by:
```
    κ(x,y) = exp(xᵀy)
```
"""
struct ExponentiatedKernel <: SimpleKernel end

kappa(κ::ExponentiatedKernel, xᵀy::Real) = exp(xᵀy)

binary_op(::ExponentiatedKernel) = DotProduct()

iskroncompatible(::ExponentiatedKernel) = true

Base.show(io::IO, ::ExponentiatedKernel) = print(io, "Exponentiated Kernel")
