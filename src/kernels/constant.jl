struct ZeroKernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::Delta
    function ZeroKernel{T,Tr}(t::Tr) where {T,Tr<:Transform}
        new{eltype{Tr},Tr}(t,Delta())
    end
end

@inline kappa(κ::ZeroKernel,d::T) where {T<:Real} = zero(T)

struct WhiteKernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::Delta
    function WhiteKernel{T,Tr}(t::Tr) where {T,Tr<:Transform}
        new{T,Tr}(t,Delta())
    end
end

function WhiteKernel()
    WhiteKernel{Float64,IdentityTransform}(IdentityTransform())
end

function WhiteKernel(t::Tr) where {Tr<:Transform}
    WhiteKernel{eltype(Tr),Tr}(t)
end

@inline kappa(κ::WhiteKernel,δₓₓ::Real) = δₓₓ


struct ConstantKernel{T,Tr<:Transform,Tc<:Real} <: Kernel{T,Tr}
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
