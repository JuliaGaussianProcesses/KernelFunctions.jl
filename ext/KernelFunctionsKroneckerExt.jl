module KernelFunctionsKroneckerExt

using KernelFunctions:
    KernelFunctions,
    Kernel,
    MOKernel,
    IndependentMOKernel,
    IntrinsicCoregionMOKernel,
    IsotopicMOInputsUnion,
    MOInputIsotopicByFeatures,
    MOInputIsotopicByOutputs,
    kernelmatrix,
    _mo_output_covariance
using Kronecker: Kronecker

# Since Kronecker does not implement `TensorCore.:⊗` but instead exports its own function
# `Kronecker.:⊗`, only the module is imported and Kronecker.:⊗ and Kronecker.kronecker are
# called explicitly.

@doc raw"""
    kernelkronmat(κ::Kernel, X::AbstractVector{<:Real}, dims::Int) -> KroneckerPower

Return a `KroneckerPower` matrix on the `D`-dimensional input grid constructed by ``\otimes_{i=1}^D X``,
where `D` is given by `dims`.

!!! warning

    Requires `Kronecker.jl` and for `iskroncompatible(κ)` to return `true`.
"""
function KernelFunctions.kernelkronmat(κ::Kernel, X::AbstractVector{<:Real}, dims::Int)
    KernelFunctions.checkkroncompatible(κ)
    K = kernelmatrix(κ, X)
    return Kronecker.kronecker(K, dims)
end

@doc raw"""

    kernelkronmat(κ::Kernel, X::AbstractVector{<:AbstractVector}) -> KroneckerProduct

Returns a `KroneckerProduct` matrix on the grid built with the collection of vectors ``\{X_i\}_{i=1}^D``: ``\otimes_{i=1}^D X_i``.

!!! warning

    Requires `Kronecker.jl` and for `iskroncompatible(κ)` to return `true`.
"""
function KernelFunctions.kernelkronmat(κ::Kernel, X::AbstractVector{<:AbstractVector})
    KernelFunctions.checkkroncompatible(κ)
    Ks = kernelmatrix.(κ, X)
    return reduce(Kronecker.:⊗, Ks)
end

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
    kronecker_kernelmatrix(
        k::Union{IndependentMOKernel,IntrinsicCoregionMOKernel}, x::MOI, y::MOI
    ) where {MOI<:IsotopicMOInputsUnion}

Requires Kronecker.jl: Computes the `kernelmatrix` for the `IndependentMOKernel` and the
`IntrinsicCoregionMOKernel`, but returns a lazy kronecker product. This object can be very
efficiently inverted or decomposed. See also [`kernelmatrix`](@ref).
"""
function KernelFunctions.kronecker_kernelmatrix(
    k::Union{IndependentMOKernel,IntrinsicCoregionMOKernel}, x::MOI, y::MOI
) where {MOI<:IsotopicMOInputsUnion}
    x.out_dim == y.out_dim ||
        throw(DimensionMismatch("`x` and `y` must have the same `out_dim`"))
    Kfeatures = kernelmatrix(k.kernel, x.x, y.x)
    Koutputs = _mo_output_covariance(k, x.out_dim)
    return _kernelmatrix_kroneckerjl_helper(MOI, Kfeatures, Koutputs)
end

function KernelFunctions.kronecker_kernelmatrix(
    k::Union{IndependentMOKernel,IntrinsicCoregionMOKernel}, x::MOI
) where {MOI<:IsotopicMOInputsUnion}
    Kfeatures = kernelmatrix(k.kernel, x.x)
    Koutputs = _mo_output_covariance(k, x.out_dim)
    return _kernelmatrix_kroneckerjl_helper(MOI, Kfeatures, Koutputs)
end

function KernelFunctions.kronecker_kernelmatrix(
    k::MOKernel, x::IsotopicMOInputsUnion, y::IsotopicMOInputsUnion
)
    return throw(
        ArgumentError("This kernel does not support a lazy kronecker kernelmatrix.")
    )
end

function KernelFunctions.kronecker_kernelmatrix(k::MOKernel, x::IsotopicMOInputsUnion)
    return KernelFunctions.kronecker_kernelmatrix(k, x, x)
end

end
