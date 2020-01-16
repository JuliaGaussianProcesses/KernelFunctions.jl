"""
`ExponentiatedKernel([ρ=1])`
The exponentiated kernel is a Mercer kernel given by:
```
    κ(x,y) = exp(ρ²xᵀy)
```
"""
struct ExponentiatedKernel{Tr} <: Kernel{Tr}
    transform::Tr
end
@inline kappa(κ::ExponentiatedKernel, xᵀy::T) where {T<:Real} = exp(xᵀy)

metric(::ExponentiatedKernel) = DotProduct()
