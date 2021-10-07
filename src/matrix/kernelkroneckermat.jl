# Since Kronecker does not implement `TensorCore.:⊗` but instead exports its own function
# `Kronecker.:⊗`, only the module is imported and Kronecker.:⊗ and Kronecker.kronecker are
# called explicitly.
using .Kronecker: Kronecker

export kernelkronmat
export kronecker_kernelmatrix

@doc raw"""
    kernelkronmat(κ::Kernel, X::AbstractVector{<:Real}, dims::Int) -> KroneckerPower

Requires `Kronecker.jl` and for `iskroncompatible(κ)` to return `true`.

Returns a `KroneckerPower` matrix on the `D`-dimensional input grid constructed by ``\otimes_{i=1}^D X``,
where `D` is given by `dims`.
"""
function kernelkronmat(κ::Kernel, X::AbstractVector{<:Real}, dims::Int)
    iskroncompatible(κ) || throw(
        ArgumentError(
            "The chosen kernel is not compatible for Kronecker matrices (see [`iskroncompatible`](@ref))",
        ),
    )
    K = kernelmatrix(κ, X)
    return Kronecker.kronecker(K, dims)
end

@doc raw"""

    kernelkronmat(κ::Kernel, X::AbstractVector{<:AbstractVector}) -> KroneckerProduct

Requires `Kronecker.jl` and for `iskroncompatible(κ)` to return `true`.

Returns a `KroneckerProduct` matrix on the grid built with the collection of vectors ``\{X_i\}_{i=1}^D``: ``\otimes_{i=1}^D X_i``.
"""
function kernelkronmat(κ::Kernel, X::AbstractVector{<:AbstractVector})
    iskroncompatible(κ) || throw(
        ArgumentError(
            "The chosen kernel is not compatible for Kronecker matrices (see [`iskroncompatible`](@ref))",
        ),
    )
    Ks = kernelmatrix.(κ, X)
    return reduce(Kronecker.:⊗, Ks)
end

@doc raw"""
    iskroncompatible(k::Kernel)

To be compatible with kroenecker constructions the kernel must satisfy
the property : for ``x,x' \in \mathbb{R}^D`
```math
k(x,x') = \prod_{i=1}^D k(x_i,x'_i)
```
Returns `false` by default.
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
    kronecker_kernelmatrix(
        k::Union{IndependentMOKernel,IntrinsicCoregionMOKernel}, x::MOI, y::MOI
    ) where {MOI<:IsotopicMOInputsUnion}

Requires Kronecker.jl: Computes the `kernelmatrix` for the `IndependentMOKernel` and the
`IntrinsicCoregionMOKernel`, but returns a lazy kronecker product. This object can be very
efficiently inverted or decomposed. See also [`kernelmatrix`](@ref).
"""
function kronecker_kernelmatrix(
    k::Union{IndependentMOKernel,IntrinsicCoregionMOKernel}, x::MOI, y::MOI
) where {MOI<:IsotopicMOInputsUnion}
    x.out_dim == y.out_dim ||
        throw(DimensionMismatch("`x` and `y` must have the same `out_dim`"))
    Kfeatures = kernelmatrix(k.kernel, x.x, y.x)
    Koutputs = _mo_output_covariance(k, x.out_dim)
    return _kernelmatrix_kroneckerjl_helper(MOI, Kfeatures, Koutputs)
end

function kronecker_kernelmatrix(
    k::Union{IndependentMOKernel,IntrinsicCoregionMOKernel}, x::MOI
) where {MOI<:IsotopicMOInputsUnion}
    Kfeatures = kernelmatrix(k.kernel, x.x)
    Koutputs = _mo_output_covariance(k, x.out_dim)
    return _kernelmatrix_kroneckerjl_helper(MOI, Kfeatures, Koutputs)
end

function kronecker_kernelmatrix(
    k::MOKernel, x::IsotopicMOInputsUnion, y::IsotopicMOInputsUnion
)
    return throw(
        ArgumentError("This kernel does not support a lazy kronecker kernelmatrix.")
    )
end

function kronecker_kernelmatrix(k::MOKernel, x::IsotopicMOInputsUnion)
    return kronecker_kernelmatrix(k, x, x)
end
