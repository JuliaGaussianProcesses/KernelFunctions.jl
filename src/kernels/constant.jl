"""
ZeroKernel([tr=IdentityTransform()])

Create a kernel always returning zero
"""
struct ZeroKernel{T,Tr} <: Kernel{T,Tr}
    transform::Tr

    function ZeroKernel{T,Tr}(t::Tr) where {T,Tr<:Transform}
        new{T,Tr}(t)
    end
end

function ZeroKernel(t::Tr=IdentityTransform()) where {Tr<:Transform}
    ZeroKernel{eltype(Tr),Tr}(t)
end

@inline kappa(κ::ZeroKernel,d::T) where {T<:Real} = zero(T)

metric(::ZeroKernel) = Delta()

"""
`WhiteKernel([tr=IdentityTransform()])`

```
    κ(x,y) = δ(x,y)
```
Kernel function working as an equivalent to add white noise.
"""
struct WhiteKernel{T,Tr} <: Kernel{T,Tr}
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
struct ConstantKernel{T,Tr,Tc<:Real} <: Kernel{T,Tr}
    transform::Tr
    c::Tc

    function ConstantKernel{T,Tr,Tc}(t::Tr,c::Tc) where {T,Tr<:Transform,Tc<:Real}
        new{T,Tr,Tc}(t,c)
    end
end

params(k::ConstantKernel) = (params(k.transform),k.c)
opt_params(k::ConstantKernel) = (opt_params(k.transform),k.c)

function ConstantKernel(c::Tc=1.0) where {Tc<:Real}
    ConstantKernel{Float64,IdentityTransform,Tc}(IdentityTransform(),c)
end

function ConstantKernel(t::Tr,c::Tc=1.0) where {Tr<:Transform,Tc<:Real}
    ConstantKernel{eltype(Tr),Tr,Tc}(t,c)
end

@inline kappa(κ::ConstantKernel,x::Real) = κ.c

metric(::ConstantKernel) = Delta()