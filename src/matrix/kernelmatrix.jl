"""
    kernelmatrix!(K::AbstractMatrix, κ::Kernel, x::AbstractVector)
    kernelmatrix!(K::AbstractMatrix, κ::Kernel, x::AbstractVector, y::AbstractVector)

In-place version of [`kernelmatrix`](@ref) where pre-allocated matrix `K` will be
overwritten with the kernel matrix.

    kernelmatrix!(K::AbstractMatrix, κ::Kernel, X::AbstractMatrix; obsdim)
    kernelmatrix!(
        K::AbstractMatrix,
        κ::Kernel,
        X::AbstractMatrix,
        Y::AbstractMatrix;
        obsdim,
    )

If `obsdim=1`, equivalent to `kernelmatrix!(K, κ, RowVecs(X))` and
`kernelmatrix(K, κ, RowVecs(X), RowVecs(Y))`, respectively.
If `obsdim=2`, equivalent to `kernelmatrix!(K, κ, ColVecs(X))` and
`kernelmatrix(K, κ, ColVecs(X), ColVecs(Y))`, respectively.

See also: [`ColVecs`](@ref), [`RowVecs`](@ref)
"""
kernelmatrix!

"""
    kernelmatrix(κ::Kernel, x::AbstractVector)

Compute the kernel `κ` for each pair of inputs in `x`.
Returns a matrix of size `(length(x), length(x))` satisfying
`kernelmatrix(κ, x)[p, q] == κ(x[p], x[q])`.

If `x` is large, consider using [`lazykernelmatrix`](@ref) instead.

    kernelmatrix(κ::Kernel, x::AbstractVector, y::AbstractVector)

Compute the kernel `κ` for each pair of inputs in `x` and `y`.
Returns a matrix of size `(length(x), length(y))` satisfying
`kernelmatrix(κ, x, y)[p, q] == κ(x[p], y[q])`.

If `x` and `y` are large, consider using [`lazykernelmatrix`](@ref) instead.

    kernelmatrix(κ::Kernel, X::AbstractMatrix; obsdim)
    kernelmatrix(κ::Kernel, X::AbstractMatrix, Y::AbstractMatrix; obsdim)

If `obsdim=1`, equivalent to `kernelmatrix(κ, RowVecs(X))` and
`kernelmatrix(κ, RowVecs(X), RowVecs(Y))`, respectively.
If `obsdim=2`, equivalent to `kernelmatrix(κ, ColVecs(X))` and
`kernelmatrix(κ, ColVecs(X), ColVecs(Y))`, respectively.

See also: [`ColVecs`](@ref), [`RowVecs`](@ref)
"""
kernelmatrix

"""
    kernelmatrix_diag!(K::AbstractVector, κ::Kernel, x::AbstractVector)
    kernelmatrix_diag!(K::AbstractVector, κ::Kernel, x::AbstractVector, y::AbstractVector)

In place version of [`kernelmatrix_diag`](@ref).

    kernelmatrix_diag!(K::AbstractVector, κ::Kernel, X::AbstractMatrix; obsdim)
    kernelmatrix_diag!(
        K::AbstractVector,
        κ::Kernel,
        X::AbstractMatrix,
        Y::AbstractMatrix;
        obsdim
    )

If `obsdim=1`, equivalent to `kernelmatrix_diag!(K, κ, RowVecs(X))` and
`kernelmatrix_diag!(K, κ, RowVecs(X), RowVecs(Y))`, respectively.
If `obsdim=2`, equivalent to `kernelmatrix_diag!(K, κ, ColVecs(X))` and
`kernelmatrix_diag!(K, κ, ColVecs(X), ColVecs(Y))`, respectively.

See also: [`ColVecs`](@ref), [`RowVecs`](@ref)
"""
kernelmatrix_diag!

"""
    kernelmatrix_diag(κ::Kernel, x::AbstractVector)

Compute the diagonal of `kernelmatrix(κ, x)` efficiently.

    kernelmatrix_diag(κ::Kernel, x::AbstractVector, y::AbstractVector)

Compute the diagonal of `kernelmatrix(κ, x, y)` efficiently.
Requires that `x` and `y` are the same length.

    kernelmatrix_diag(κ::Kernel, X::AbstractMatrix; obsdim)
    kernelmatrix_diag(κ::Kernel, X::AbstractMatrix, Y::AbstractMatrix; obsdim)

If `obsdim=1`, equivalent to `kernelmatrix_diag(κ, RowVecs(X))` and
`kernelmatrix_diag(κ, RowVecs(X), RowVecs(Y))`, respectively.
If `obsdim=2`, equivalent to `kernelmatrix_diag(κ, ColVecs(X))` and
`kernelmatrix_diag(κ, ColVecs(X), ColVecs(Y))`, respectively.

See also: [`ColVecs`](@ref), [`RowVecs`](@ref)
"""
kernelmatrix_diag

#
# Kernel implementations. Generic fallbacks that depend only on kernel evaluation.
#

kernelmatrix!(K::AbstractMatrix, κ::Kernel, x::AbstractVector) = kernelmatrix!(K, κ, x, x)

function kernelmatrix!(K::AbstractMatrix, κ::Kernel, x::AbstractVector, y::AbstractVector)
    validate_inplace_dims(K, x, y)
    K .= κ.(x, permutedims(y))
    return K
end

kernelmatrix(κ::Kernel, x::AbstractVector) = kernelmatrix(κ, x, x)

function kernelmatrix(κ::Kernel, x::AbstractVector, y::AbstractVector)
    validate_inputs(x, y)
    return κ.(x, permutedims(y))
end

function kernelmatrix_diag!(K::AbstractVector, κ::Kernel, x::AbstractVector)
    validate_inplace_dims(K, x)
    return map!(x -> κ(x, x), K, x)
end

function kernelmatrix_diag!(
    K::AbstractVector, κ::Kernel, x::AbstractVector, y::AbstractVector
)
    return map!(κ, K, x, y)
end

kernelmatrix_diag(κ::Kernel, x::AbstractVector) = map(x -> κ(x, x), x)

kernelmatrix_diag(κ::Kernel, x::AbstractVector, y::AbstractVector) = map(κ, x, y)

#
# SimpleKernel optimisations.
#

function kernelmatrix!(K::AbstractMatrix, κ::SimpleKernel, x::AbstractVector)
    validate_inplace_dims(K, x)
    pairwise!(K, metric(κ), x)
    return map!(x -> kappa(κ, x), K, K)
end

function kernelmatrix!(
    K::AbstractMatrix, κ::SimpleKernel, x::AbstractVector, y::AbstractVector
)
    validate_inplace_dims(K, x, y)
    pairwise!(K, metric(κ), x, y)
    return map!(x -> kappa(κ, x), K, K)
end

function kernelmatrix(κ::SimpleKernel, x::AbstractVector)
    return map(x -> kappa(κ, x), pairwise(metric(κ), x))
end

function kernelmatrix(κ::SimpleKernel, x::AbstractVector, y::AbstractVector)
    validate_inputs(x, y)
    return map(x -> kappa(κ, x), pairwise(metric(κ), x, y))
end

function kernelmatrix_diag(κ::SimpleKernel, x::AbstractVector)
    return map(x -> kappa(κ, x), colwise(metric(κ), x))
end

function kernelmatrix_diag(κ::SimpleKernel, x::AbstractVector, y::AbstractVector)
    return map(x -> kappa(κ, x), colwise(metric(κ), x, y))
end

#
# Wrapper methods for AbstractMatrix inputs to maintain obsdim interface.
#

const defaultobs = nothing

function kernelmatrix!(
    K::AbstractMatrix, κ::Kernel, X::AbstractMatrix; obsdim::Union{Int,Nothing}=defaultobs
)
    return kernelmatrix!(K, κ, vec_of_vecs(X; obsdim=obsdim))
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::Kernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Union{Int,Nothing}=defaultobs,
)
    return kernelmatrix!(K, κ, vec_of_vecs(X; obsdim=obsdim), vec_of_vecs(Y; obsdim=obsdim))
end

function kernelmatrix(κ::Kernel, X::AbstractMatrix; obsdim::Union{Int,Nothing}=defaultobs)
    return kernelmatrix(κ, vec_of_vecs(X; obsdim=obsdim))
end

function kernelmatrix(κ::Kernel, X::AbstractMatrix, Y::AbstractMatrix; obsdim=defaultobs)
    return kernelmatrix(κ, vec_of_vecs(X; obsdim=obsdim), vec_of_vecs(Y; obsdim=obsdim))
end

function kernelmatrix_diag!(
    K::AbstractVector, κ::Kernel, X::AbstractMatrix; obsdim::Union{Int,Nothing}=defaultobs
)
    return kernelmatrix_diag!(K, κ, vec_of_vecs(X; obsdim=obsdim))
end

function kernelmatrix_diag!(
    K::AbstractVector,
    κ::Kernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Union{Int,Nothing}=defaultobs,
)
    return kernelmatrix_diag!(
        K, κ, vec_of_vecs(X; obsdim=obsdim), vec_of_vecs(Y; obsdim=obsdim)
    )
end

function kernelmatrix_diag(
    κ::Kernel, X::AbstractMatrix; obsdim::Union{Int,Nothing}=defaultobs
)
    return kernelmatrix_diag(κ, vec_of_vecs(X; obsdim=obsdim))
end

function kernelmatrix_diag(
    κ::Kernel, X::AbstractMatrix, Y::AbstractMatrix; obsdim::Union{Int,Nothing}=defaultobs
)
    return kernelmatrix_diag(
        κ, vec_of_vecs(X; obsdim=obsdim), vec_of_vecs(Y; obsdim=obsdim)
    )
end
