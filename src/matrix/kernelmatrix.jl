"""
    kernelmatrix!(K::AbstractMatrix, κ::Kernel, X; obsdim::Integer = 2)
    kernelmatrix!(K::AbstractMatrix, κ::Kernel, X, Y; obsdim::Integer = 2)

In-place version of [`kernelmatrix`](@ref) where pre-allocated matrix `K` will be
overwritten with the kernel matrix.
"""
kernelmatrix!

"""
    kernelmatrix(κ::Kernel, X; obsdim::Int = 2)
    kernelmatrix(κ::Kernel, X, Y; obsdim::Int = 2)

Calculate the kernel matrix of `X` (and `Y`) with respect to kernel `κ`.
`obsdim = 1` means the matrix `X` (and `Y`) has size #samples x #dimension
`obsdim = 2` means the matrix `X` (and `Y`) has size #dimension x #samples
"""
kernelmatrix

"""
    kerneldiagmatrix!(K::AbstractVector, κ::Kernel, X; obsdim::Int = 2)
    kerneldiagmatrix!(K::AbstractVector, κ::Kernel, X, Y; obsdim::Int = 2)

In place version of [`kerneldiagmatrix`](@ref)
"""
kerneldiagmatrix!

"""
    kerneldiagmatrix(κ::Kernel, X; obsdim::Int = 2)

Calculate the diagonal matrix of `X` with respect to kernel `κ`
`obsdim = 1` means the matrix `X` has size #samples x #dimension
`obsdim = 2` means the matrix `X` has size #dimension x #samples

    kerneldiagmatrix(κ::Kernel, X, Y; obsdim::Int = 2)

Calculate the diagonal of `kernelmatrix(κ, X, Y; obsdim)` efficiently. Requires that `X` and
`Y` are the same length.
"""
kerneldiagmatrix



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

function kerneldiagmatrix!(K::AbstractVector, κ::Kernel, x::AbstractVector)
    validate_inplace_dims(K, x)
    return map!(x -> κ(x, x), K, x)
end

function kerneldiagmatrix!(
    K::AbstractVector, κ::Kernel, x::AbstractVector, y::AbstractVector,
)
    return map!(κ, x, y)
end

kerneldiagmatrix(κ::Kernel, x::AbstractVector) = map(x -> κ(x, x), x)

kerneldiagmatrix(κ::Kernel, x::AbstractVector, y::AbstractVector) = map(κ, x, y)



#
# SimpleKernel optimisations.
#

function kernelmatrix!(K::AbstractMatrix, κ::SimpleKernel, x::AbstractVector)
    validate_inplace_dims(K, x)
    pairwise!(K, binary_op(κ), x)
    return map!(d -> kappa(κ, d), K, K)
end

function kernelmatrix!(
    K::AbstractMatrix, κ::SimpleKernel, x::AbstractVector, y::AbstractVector,
)
    validate_inplace_dims(K, x, y)
    pairwise!(K, binary_op(κ), x, y)
    return map!(d -> kappa(κ, d), K, K)
end

function kernelmatrix(κ::SimpleKernel, x::AbstractVector)
    return map(d -> kappa(κ, d), pairwise(binary_op(κ), x))
end

function kernelmatrix(κ::SimpleKernel, x::AbstractVector, y::AbstractVector)
    validate_inputs(x, y)
    return map(d -> kappa(κ, d), pairwise(binary_op(κ), x, y))
end



#
# Wrapper methods for AbstractMatrix inputs to maintain obsdim interface.
#

const defaultobs = 2

function kernelmatrix!(
    K::AbstractMatrix, κ::Kernel, X::AbstractMatrix; obsdim::Int=defaultobs,
)
    return kernelmatrix!(K, κ, vec_of_vecs(X; obsdim=obsdim))
end

function kernelmatrix!(
    K::AbstractMatrix, κ::Kernel, X::AbstractMatrix, Y::AbstractMatrix;
    obsdim::Int=defaultobs,
)
    return kernelmatrix!(K, κ, vec_of_vecs(X; obsdim=obsdim), vec_of_vecs(Y; obsdim=obsdim))
end

function kernelmatrix(κ::Kernel, X::AbstractMatrix; obsdim::Int=defaultobs)
    return kernelmatrix(κ, vec_of_vecs(X; obsdim=obsdim))
end

function kernelmatrix(κ::Kernel, X::AbstractMatrix, Y::AbstractMatrix; obsdim=defaultobs)
    return kernelmatrix(κ, vec_of_vecs(X; obsdim=obsdim), vec_of_vecs(Y; obsdim=obsdim))
end

function kerneldiagmatrix!(
    K::AbstractVector, κ::Kernel, X::AbstractMatrix; obsdim::Int=defaultobs
)
    return kerneldiagmatrix!(K, κ, vec_of_vecs(X; obsdim=obsdim))
end

function kerneldiagmatrix!(
    K::AbstractVector, κ::Kernel, X::AbstractMatrix, Y::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    return kerneldiagmatrix!(
        K, κ, vec_of_vecs(X; obsdim=obsdim), vec_of_vecs(Y; obsdim=obsdim),
    )
end

function kerneldiagmatrix(κ::Kernel, X::AbstractMatrix; obsdim::Int=defaultobs)
    return kerneldiagmatrix(κ, vec_of_vecs(X; obsdim=obsdim))
end

function kerneldiagmatrix(
    κ::Kernel, X::AbstractMatrix, Y::AbstractMatrix; obsdim::Int=defaultobs,
)
    return kerneldiagmatrix(κ, vec_of_vecs(X; obsdim=obsdim), vec_of_vecs(Y; obsdim=obsdim))
end
