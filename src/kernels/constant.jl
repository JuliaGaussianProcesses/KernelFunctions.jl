"""
ZeroKernel([tr=IdentityTransform()])

Create a kernel always returning zero
"""
struct ZeroKernel{T,Tr} <: Kernel{Tr}
    transform::Tr

    function ZeroKernel{T,Tr}(t::Tr) where {T,Tr<:Transform}
        new{T,Tr}(t)
    end
end

function ZeroKernel(t::Tr=IdentityTransform()) where {Tr<:Transform}
    ZeroKernel{eltype(Tr),Tr}(t)
end

@inline kappa(κ::ZeroKernel, d::T) where {T<:Real} = zero(T)

metric(::ZeroKernel) = Delta()

"""
`WhiteKernel([tr=IdentityTransform()])`

```
    κ(x,y) = δ(x,y)
```
Kernel function working as an equivalent to add white noise.
"""
struct WhiteKernel{T,Tr} <: Kernel{Tr}
    transform::Tr

    function WhiteKernel{T,Tr}(t::Tr) where {T,Tr<:Transform}
        new{T,Tr}(t)
    end
end

function WhiteKernel(t::Tr=IdentityTransform()) where {Tr<:Transform}
    WhiteKernel{eltype(Tr),Tr}(t)
end

@inline kappa(κ::WhiteKernel,δₓₓ::Real) = δₓₓ

metric(::WhiteKernel) = Delta()

"""
`ConstantKernel([tr=IdentityTransform(),[c=1.0]])`
```
    κ(x,y) = c
```
Kernel function always returning a constant value `c`
"""
struct ConstantKernel{Tr, Tc<:Real} <: Kernel{Tr}
    transform::Tr
    c::Tc
end

params(k::ConstantKernel) = (params(k.transform),k.c)
opt_params(k::ConstantKernel) = (opt_params(k.transform),k.c)

ConstantKernel(c::Real=1.0) = ConstantKernel(IdentityTransform(),c)

ConstantKernel(t::Tr,c::Tc=1.0) where {Tr<:Transform,Tc<:Real} = ConstantKernel{Tr,Tc}(t,c)

@inline kappa(κ::ConstantKernel,x::Real) = κ.c

metric(::ConstantKernel) = Delta()
