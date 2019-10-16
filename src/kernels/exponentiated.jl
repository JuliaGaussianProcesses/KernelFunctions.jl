"""
    ExponentiatedKernel([ρ=1])

    The exponentiated kernel is a Mercer kernel given by:

```
        κ(x,y) = exp(xᵀy)
```
"""
struct ExponentiatedKernel{T,Tr} <: Kernel{T,Tr}
    transform::Tr
    metric::DotProduct
    function ExponentiatedKernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform,DotProduct())
    end
end
@inline kappa(κ::ExponentiatedKernel, xᵀy::T) where {T<:Real} = exp(xᵀy)
