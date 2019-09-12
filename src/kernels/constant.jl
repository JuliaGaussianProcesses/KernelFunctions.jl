struct ZeroKernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::Delta
    function ZeroKernel{T,Tr}(t::Tr) where {Tr<:Transform}
        new{eltype{Tr},Tr}(t,Delta())
    end
end

@inline kappa(κ::ZeroKernel,d::T) where {T<:Real} = zero(T)

struct WhiteKernel{T,Tr<:Transform} <: Kernel{T,Tr}
    transform::Tr
    metric::Delta
    function WhiteKernel{T,Tr}(t::Tr) where {Tr<:Transform}
        new{T,Transform}(t,Delta())
    end
end

function WhiteKernel()
    WhiteKernel{Float64,IdentityTransform}(IdentityTransform())
end

function WhiteKernel(t::Tr) where {Tr<:Transform}
    WhiteKernel{eltype(Tr),Tr}(t)
end

@inline kappa(κ::WhiteKernel,δₓₓ::Real) = δₓₓ
