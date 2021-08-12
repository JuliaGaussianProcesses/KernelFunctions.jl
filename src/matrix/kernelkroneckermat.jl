# Since Kronecker does not implement `TensorCore.:⊗` but instead exports its own function
# `Kronecker.:⊗`, only the module is imported and Kronecker.:⊗ and Kronecker.kronecker are
# called explicitly.
using .Kronecker: Kronecker

export kernelkronmat

function kernelkronmat(κ::Kernel, X::AbstractVector, dims::Int)
    @assert iskroncompatible(κ) "The chosen kernel is not compatible for kroenecker matrices (see [`iskroncompatible`](@ref))"
    k = kernelmatrix(κ, X)
    return Kronecker.kronecker(k, dims)
end

function kernelkronmat(
    κ::Kernel, X::AbstractVector{<:AbstractVector}; obsdim::Int=defaultobs
)
    @assert iskroncompatible(κ) "The chosen kernel is not compatible for Kronecker matrices"
    Ks = kernelmatrix.(κ, X)
    return K = reduce(Kronecker.:⊗, Ks)
end

"""
    To be compatible with kroenecker constructions the kernel must satisfy
    the property : for x,x' ∈ ℜᴰ
    k(x,x') = ∏ᵢᴰ k(xᵢ,x'ᵢ)
"""
@inline iskroncompatible(κ::Kernel) = false # Default return for kernels

@inline ismatrixkroncompatible(κ::MOKernel) = false # Default return for kernels
@inline ismatrixkroncompatible(κ::IndependentMOKernel) = true
@inline ismatrixkroncompatible(κ::IntrinsicCoregionMOKernel) = true

function _kroneckerkernelmatrix(Ktmp, B, ::MOInputIsotopicByFeatures)
    return Kronecker.kronecker(Ktmp, B)
end

function _kroneckerkernelmatrix(Ktmp, B, ::MOInputIsotopicByOutputs)
    return Kronecker.kronecker(B, Ktmp)
end

function kernelkronmat(k::IndependentMOKernel, x::MOI, y::MOI) where {MOI<:IsotopicMOInputsUnion}
    @assert x.out_dim == y.out_dim
    Ktmp = kernelmatrix(k.kernel, x.x, y.x)
    mtype = eltype(Ktmp)
    return _kroneckerkernelmatrix(Ktmp, Eye{mtype}(x.out_dim), x)
end

function kernelkronmat(
    k::IntrinsicCoregionMOKernel, x::MOI, y::MOI
) where {MOI<:IsotopicMOInputsUnion}
    @assert x.out_dim == y.out_dim
    Ktmp = kernelmatrix(k.kernel, x.x, y.x)
    return _kroneckerkernelmatrix(Ktmp, k.B, x)
end

function kernelkronmat(k::MOK, x::MOI) where {MOI<:IsotopicMOInputsUnion,MOK<:MOKernel}
    @assert iskroncompatible(κ) "The chosen kernel is not compatible for Kronecker matrices"
    return kernelkronmat(k, x, x)
end
