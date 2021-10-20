# Since Kronecker does not implement `TensorCore.:⊗` but instead exports its own function
# `Kronecker.:⊗`, only the module is imported and Kronecker.:⊗ and Kronecker.kronecker are
# called explicitly.
using .Kronecker: Kronecker

export kernelkronmat

@doc raw"""
    kernelkronmat(k::Kernel, x::AbstractVector{<:Real}, d::Int)

Compute the [`kernelmatrix`](@ref) for kernel `k` on the `d`-dimensional grid ``\otimes_{i=1}^d x``
as a lazy kronecker product.

!!! warning
    You have to load [Kronecker.jl](https://github.com/MichielStock/Kronecker.jl) to use this function.
    Additionally, `iskroncompatible(k)` has to be `true`.
"""
function kernelkronmat(k::Kernel, x::AbstractVector{<:Real}, d::Int)
    checkkroncompatible(k)
    K = kernelmatrix(k, x)
    return Kronecker.kronecker(K, d)
end

@doc raw"""
    kernelkronmat(k::Kernel, x::AbstractVector{<:AbstractVector})

Compute the [`kernelmatrix`](@ref) for kernel `k` on the grid ``\otimes_{i} x_i`` as a lazy kronecker product.

!!! warning
    You have to load [Kronecker.jl](https://github.com/MichielStock/Kronecker.jl) to use this function.
    Additionally, `iskroncompatible(k)` has to be `true`.
"""
function kernelkronmat(k::Kernel, x::AbstractVector{<:AbstractVector})
    checkkroncompatible(k)
    Ks = kernelmatrix.(k, x)
    return reduce(Kronecker.:⊗, Ks)
end

@doc raw"""
    iskroncompatible(k::Kernel)

Determine whether kernel `k` is compatible with Kronecker constructions such as [`kernelkronmat`](@ref).

The function returns `false` by default. If `k` is compatible it must satisfy for all ``x, x' \in \mathbb{R}^d`:
```math
k(x, x') = \prod_{i=1}^d k(x_i, x'_i).
```
"""
@inline iskroncompatible(::Kernel) = false # Default return for kernels

function checkkroncompatible(k::Kernel)
    return iskroncompatible(k) || throw(
        ArgumentError(
            "the chosen kernel is not compatible for Kronecker matrices (see [`iskroncompatible`](@ref))",
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

Compute the [`kernelmatrix`](@ref) for kernel `k` with inputs `x` and `y` as a lazy kronecker product.

The returned kernel matrix can be inverted or decomposed efficiently.

!!! warning
    You have to load [Kronecker.jl](https://github.com/MichielStock/Kronecker.jl) to use this function.
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
            "computation of the kernel matrix as a lazy kronecker product is not " *
            "supported for the given kernel and inputs",
        ),
    )
end

# deprecations
Base.@deprecate kronecker_kernelmatrix(k::MOKernel, x::IsotopicMOInputsUnion) kernelmatrix(
    Kronecker.KroneckerProduct, k, x,
)
Base.@deprecate kronecker_kernelmatrix(
    k::MOKernel, x::IsotopicMOInputsUnion, y::IsotopicMOInputsUnion
) kernelmatrix(Kronecker.KroneckerProduct, k, x, y)
