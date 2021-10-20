# Since Kronecker does not implement `TensorCore.:⊗` but instead exports its own function
# `Kronecker.:⊗`, only the module is imported and Kronecker.:⊗ and Kronecker.kronecker are
# called explicitly.
using .Kronecker: Kronecker

export kernelkronmat

@doc raw"""
    kernelkronmat(κ::Kernel, X::AbstractVector{<:Real}, dims::Int) -> KroneckerPower

Return a `KroneckerPower` matrix on the `D`-dimensional input grid constructed by ``\otimes_{i=1}^D X``,
where `D` is given by `dims`.

!!! warning

    Require `Kronecker.jl` and for `iskroncompatible(κ)` to return `true`.
"""
function kernelkronmat(κ::Kernel, X::AbstractVector{<:Real}, dims::Int)
    checkkroncompatible(κ)
    K = kernelmatrix(κ, X)
    return Kronecker.kronecker(K, dims)
end

@doc raw"""

    kernelkronmat(κ::Kernel, X::AbstractVector{<:AbstractVector}) -> KroneckerProduct

Returns a `KroneckerProduct` matrix on the grid built with the collection of vectors ``\{X_i\}_{i=1}^D``: ``\otimes_{i=1}^D X_i``.

!!! warning

    Requires `Kronecker.jl` and for `iskroncompatible(κ)` to return `true`.
"""
function kernelkronmat(κ::Kernel, X::AbstractVector{<:AbstractVector})
    checkkroncompatible(κ)
    Ks = kernelmatrix.(κ, X)
    return reduce(Kronecker.:⊗, Ks)
end

@doc raw"""
    iskroncompatible(k::Kernel)

Determine whether kernel `k` is compatible with Kronecker constructions such as [`kernelkronmat`](@ref)

The function returns `false` by default. If `k` is compatible it must satisfy for all ``x, x' \in \mathbb{R}^D`:
```math
k(x, x') = \prod_{i=1}^D k(x_i, x'_i).
```
"""
@inline iskroncompatible(κ::Kernel) = false # Default return for kernels

function checkkroncompatible(κ::Kernel)
    return iskroncompatible(κ) || throw(
        ArgumentError(
            "The chosen kernel is not compatible for Kronecker matrices (see [`iskroncompatible`](@ref))",
        ),
    )
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
