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

function _kernelmatrix_kroneckerjl_helper(
    ::Type{<:MOInputIsotopicByFeatures}, Kfeatures, Koutputs
)
    return Kronecker.kronecker(Kfeatures, Koutputs)
end

function _kernelmatrix_kroneckerjl_helper(
    ::Type{<:MOInputIsotopicByOutputs}, Kfeatures, Koutputs
)
    return Kronecker.kronecker(Koutputs, Kfeatures)
end

"""
    kernelmatrix(
        ::Type{<:Kronecker.KroneckerProduct},
        k::Union{IndependentMOKernel,IntrinsicCoregionMOKernel},
        x::MOI,
        y::MOI,
    ) where {MOI<:IsotopicMOInputsUnion}

Compute the `kernelmatrix` for the `IndependentMOKernel` and the `IntrinsicCoregionMOKernel`
as a lazy kronecker product.

The returned kernel matrix can be inverted or decomposed efficiently.

!!! note
    You have to load Kronecker.jl if you would like to use this function.
"""
function kernelmatrix(
    ::Type{T}, k::Union{IndependentMOKernel,IntrinsicCoregionMOKernel}, x::MOI, y::MOI
)::T where {T<:Kronecker.KroneckerProduct,MOI<:IsotopicMOInputsUnion}
    x.out_dim == y.out_dim ||
        throw(DimensionMismatch("`x` and `y` must have the same `out_dim`"))
    Kfeatures = kernelmatrix(k.kernel, x.x, y.x)
    Koutputs = _mo_output_covariance(k, x.out_dim)
    return _kernelmatrix_kroneckerjl_helper(MOI, Kfeatures, Koutputs)
end

function kernelmatrix(
    ::Type{T}, k::Union{IndependentMOKernel,IntrinsicCoregionMOKernel}, x::MOI
)::T where {T<:Kronecker.KroneckerProduct,MOI<:IsotopicMOInputsUnion}
    Kfeatures = kernelmatrix(k.kernel, x.x)
    Koutputs = _mo_output_covariance(k, x.out_dim)
    return _kernelmatrix_kroneckerjl_helper(MOI, Kfeatures, Koutputs)
end

function kernelmatrix(
    ::Type{<:Kronecker.KroneckerProduct}, k::Kernel, x::AbstractVector, y::AbstractVector=x
)
    return throw(
        ArgumentError(
            "computation of the kernel matrix as a lazy kronecker matrix is not " *
            "supported for the given kernel and inputs",
        ),
    )
end

# deprecations
Base.@deprecate kronecker_kernelmatrix(
    k::MOKernel, x::IsotopicMOInputsUnion, y::IsotopicMOInputsUnion=x
) kernelmatrix(Kronecker.KroneckerProduct, k, x, y)
