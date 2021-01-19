"""
    ExponentiatedKernel()

Exponentiated kernel.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the exponentiated kernel is defined as
```math
k(x, x') = \\exp(x^\\top x').
```
"""
struct ExponentiatedKernel <: SimpleKernel end

kappa(::ExponentiatedKernel, xᵀy::Real) = exp(xᵀy)

metric(::ExponentiatedKernel) = DotProduct()

iskroncompatible(::ExponentiatedKernel) = true

Base.show(io::IO, ::ExponentiatedKernel) = print(io, "Exponentiated Kernel")
