"""
`ExponentiatedKernel([ρ=1])`
The exponentiated kernel is a Mercer kernel given by:
```
    κ(x,y) = exp(ρ²xᵀy)
```
"""
struct ExponentiatedKernel <: BaseKernel end

kappa(κ::ExponentiatedKernel, xᵀy::Real) = exp(xᵀy)

metric(::ExponentiatedKernel) = DotProduct()

iskroncompatible(::ExponentiatedKernel) = true
