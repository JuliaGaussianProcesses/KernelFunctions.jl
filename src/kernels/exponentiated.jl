"""
`ExponentiatedKernel([ρ=1])`
The exponentiated kernel is a Mercer kernel given by:
```
    κ(x,y) = exp(ρ²xᵀy)
```
"""
struct ExponentiatedKernel <: Kernel end

@inline kappa(κ::ExponentiatedKernel, xᵀy::Real) = exp(xᵀy)

metric(::ExponentiatedKernel) = DotProduct()

@inline iskroncompatible(::ExponentiatedKernel) = true
