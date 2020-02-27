using .Kronecker

export kernelkronmat

function kernelkronmat(
    κ::Kernel,
    X::AbstractVector,
    dims::Int
    )
    @assert iskroncompatible(κ) "The chosen kernel is not compatible for kroenecker matrices (see `iskroncompatible()`)"
    k = kernelmatrix(κ,reshape(X,:,1),obsdim=1)
    kronecker(k,dims)
end

function kernelkronmat(
    κ::Kernel,
    X::AbstractVector{<:AbstractVector};
    obsdim::Int=defaultobs
    )
    @assert iskroncompatible(κ) "The chosen kernel is not compatible for kroenecker matrices"
    Ks = kernelmatrix.(κ,X,obsdim=obsdim)
    K = reduce(⊗,Ks)
end


"""
    To be compatible with kroenecker constructions the kernel must satisfy
    the property : for x,x' ∈ ℜᴰ
    k(x,x') = ∏ᵢᴰ k(xᵢ,x'ᵢ)
"""
@inline iskroncompatible(κ::Kernel) = false # Default return for kernels
