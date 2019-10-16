"""
    ZeroKernel()

    Create a kernel that always return a zero kernel matrix

"""
struct ZeroKernel{T,Tr} <: Kernel{T,Tr}
    transform::Tr
    metric::Delta
    function ZeroKernel{T,Tr}(t::Tr) where {T,Tr<:Transform}
        new{T,Tr}(t,Delta())
    end
end

function ZeroKernel(t::Tr=IdentityTransform()) where {Tr<:Transform}
    ZeroKernel{eltype(Tr),Tr}(t)
end

@inline kappa(κ::ZeroKernel,d::T) where {T<:Real} = zero(T)

"""
    WhiteKernel()

```
    κ(x,y) = δ(x,y)
```
    Kernel function working as an equivalent to add white noise.
"""
struct WhiteKernel{T,Tr} <: Kernel{T,Tr}
    transform::Tr
    metric::Delta
    function WhiteKernel{T,Tr}(t::Tr) where {T,Tr<:Transform}
        new{T,Tr}(t,Delta())
    end
end

function WhiteKernel(t::Tr=IdentityTransform()) where {Tr<:Transform}
    WhiteKernel{eltype(Tr),Tr}(t)
end

@inline kappa(κ::WhiteKernel,δₓₓ::Real) = δₓₓ

"""
    ConstantKernel([c=1.0])

```
    κ(x,y) = c
```
    Kernel function always returning a constant value `c`
"""
struct ConstantKernel{T,Tr,Tc<:Real} <: Kernel{T,Tr}
    transform::Tr
    metric::Delta
    c::Tc
    function ConstantKernel{T,Tr,Tc}(t::Tr,c::Tc) where {T,Tr<:Transform,Tc<:Real}
        new{T,Tr,Tc}(t,Delta(),c)
    end
end

function ConstantKernel(c::Tc=1.0) where {Tc<:Real}
    ConstantKernel{Float64,IdentityTransform,Tc}(IdentityTransform(),c)
end

function ConstantKernel(t::Tr,c::Tc=1.0) where {Tr<:Transform,Tc<:Real}
    ConstantKernel{eltype(Tr),Tr,Tc}(t,c)
end

@inline kappa(κ::ConstantKernel,x::Real) = κ.c
