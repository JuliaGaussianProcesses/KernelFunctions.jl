using .Kronecker

export kernelkronmat

function kernelkronmat(κ::Kernel, X::AbstractVector, dims::Int)
    @assert iskroncompatible(κ) "The chosen kernel is not compatible for kroenecker matrices (see [`iskroncompatible`](@ref))"
    k = kernelmatrix(κ, X)
    return kronecker(k, dims)
end

function kernelkronmat(
    κ::Kernel, X::AbstractVector{<:AbstractVector}; obsdim::Int=defaultobs
)
    @assert iskroncompatible(κ) "The chosen kernel is not compatible for Kronecker matrices"
    Ks = kernelmatrix.(κ, X)
    return K = reduce(⊗, Ks)
end

"""
    To be compatible with kroenecker constructions the kernel must satisfy
    the property : for x,x' ∈ ℜᴰ
    k(x,x') = ∏ᵢᴰ k(xᵢ,x'ᵢ)
"""
@inline iskroncompatible(κ::Kernel) = false # Default return for kernels
