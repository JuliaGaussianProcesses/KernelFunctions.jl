"""
`ExponentiatedKernel([ρ=1])`
The exponentiated kernel is a Mercer kernel given by:
```
    κ(x,y) = exp(ρ²xᵀy)
```
"""
struct ExponentiatedKernel{T,Tr} <: Kernel{T,Tr}
    transform::Tr

    function ExponentiatedKernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform)
    end
end
@inline kappa(κ::ExponentiatedKernel, xᵀy::T) where {T<:Real} = exp(xᵀy)

metric(::ExponentiatedKernel) = DotProduct()