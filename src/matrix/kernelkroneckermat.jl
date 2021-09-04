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

function _kernelmatrix_kroneckerjl_helper(::MOInputIsotopicByFeatures, Kfeatures, Koutputs)
    return Kronecker.kronecker(Kfeatures, Koutputs)
end

function _kernelmatrix_kroneckerjl_helper(::MOInputIsotopicByOutputs, Kfeatures, Koutputs)
    return Kronecker.kronecker(Koutputs, Kfeatures)
end

_mo_output_covariance(k::IndependentMOKernel, out_dim) = Eye{Bool}(out_dim)
function _mo_output_covariance(k::IntrinsicCoregionMOKernel, out_dim)
    @assert size(k.B) == (out_dim, out_dim)
    return k.B
end

function kernelmatrix(
    k::IndependentMOKernel,
    x::IsotopicMOInputsUnion,
    y::IsotopicMOInputsUnion,
)
    @assert x.out_dim == y.out_dim
    Kfeatures = kernelmatrix(k.kernel, x.x, y.x)
    Koutputs = Eye{Bool}(x.out_dim)
    return _kernelmatrix_kroneckerjl_helper(x, Kfeatures, Koutputs)
end

function kernelmatrix(
    k::IntrinsicCoregionMOKernel,
    x::IsotopicMOInputsUnion,
    y::IsotopicMOInputsUnion,
)
    @assert x.out_dim == y.out_dim
    Kfeatures = kernelmatrix(k.kernel, x.x, y.x)
    Koutputs = k.B
    return _kernelmatrix_kroneckerjl_helper(x, Kfeatures, Koutputs)
end

# function kernelmatrix(
#     k::Union{IndependentMOKernel,IntrinsicCoregionMOKernel},
#     x::IsotopicMOInputsUnion,
#     y::IsotopicMOInputsUnion,
# )
#     @assert x.out_dim == y.out_dim
#     Kfeatures = kernelmatrix(k.kernel, x.x, y.x)
#     Koutputs = _mo_output_covariance(k, x.out_dim)
#     return _kernelmatrix_kroneckerjl_helper(Kfeatures, Koutputs, x)
# end

function kernelmatrix(
    k::Union{IndependentMOKernel,IntrinsicCoregionMOKernel}, x::IsotopicMOInputsUnion
)
    Kfeatures = kernelmatrix(k.kernel, x.x)
    Koutputs = _mo_output_covariance(k, x.out_dim)
    return _kernelmatrix_kroneckerjl_helper(Kfeatures, Koutputs, x)
end

# function kronecker_kernelmatrix(
#     k::MOKernel, x::IsotopicMOInputsUnion, y::IsotopicMOInputsUnion
# )
#     return throw(
#         ArgumentError("This kernel does not support a lazy kronecker kernelmatrix.")
#     )
# end

# function kronecker_kernelmatrix(k::MOKernel, x::IsotopicMOInputsUnion)
#     return kronecker_kernelmatrix(k, x, x)
# end
