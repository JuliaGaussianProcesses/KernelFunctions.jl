"""
    kernelmatrix!(K::AbstractMatrix, κ::Kernel, X; obsdim::Integer = 2)
    kernelmatrix!(K::AbstractMatrix, κ::Kernel, X, Y; obsdim::Integer = 2)

In-place version of [`kernelmatrix`](@ref) where pre-allocated matrix `K` will be overwritten with the kernel matrix.
"""
kernelmatrix!

## Wrapper for vector of reals
function kernelmatrix!(
    K::AbstractMatrix,
    κ::Kernel,
    X::AbstractVector{<:Real}
)
    kernelmatrix!(K, κ, reshape(X, 1, :), obsdim = 2)
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::SimpleKernel,
    X::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    @assert obsdim ∈ [1, 2] "obsdim should be 1 or 2 (see docs of `kernelmatrix`)"
    if !check_dims(K, X, X, feature_dim(obsdim), obsdim)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
    end
    map!(x -> kappa(κ, x), K, pairwise(metric(κ), X, dims = obsdim))
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::Kernel,
    X::AbstractMatrix;
    obsdim::Int = defaultobs
)
    @assert obsdim ∈ [1, 2] "obsdim should be 1 or 2 (see docs of `kernelmatrix`))"
    if obsdim == 1
        kernelmatrix!(K, κ, ColVecs(X'))
    else
        kernelmatrix!(K, κ, ColVecs(X))
    end
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::Kernel,
    X::AbstractVector
    )
    if !check_dims(K, X, X, feature_dim(obsdim), obsdim)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
    end
    map!(κ, K, X, X')
end

## Wrapper for vector of reals
function kernelmatrix!(
    K::AbstractMatrix,
    κ::Kernel,
    X::AbstractVector{<:Real},
    Y::AbstractVector{<:Real}
)
    kernelmatrix!(K, κ, reshape(X, 1, :), reshape(Y, 1, :), obsdim = 2)
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::SimpleKernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    @assert obsdim ∈ [1, 2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    if !check_dims(K, X, Y, feature_dim(obsdim), obsdim)
        throw(DimensionMismatch("Dimensions $(size(K)) of the target array K are not consistent with X ($(size(X))) and Y ($(size(Y)))"))
    end
    map!(x -> kappa(κ, x), K, pairwise(metric(κ), X, Y, dims = obsdim))
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::Kernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs
)
    @assert obsdim ∈ [1, 2] "obsdim should be 1 or 2 (see docs of `kernelmatrix`))"
    if obsdim == 1
        kernelmatrix!(K, κ, ColVecs(X'), ColVecs(Y'))
    else
        kernelmatrix!(K, κ, ColVecs(X), ColVecs(Y))
    end
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::Kernel,
    X::AbstractVector,
    Y::AbstractVector
    )
    map!(K, κ, X, Y')
end

"""
    kernelmatrix(κ::Kernel, X; obsdim::Int = 2)
    kernelmatrix(κ::Kernel, X, Y; obsdim::Int = 2)

Calculate the kernel matrix of `X` (and `Y`) with respect to kernel `κ`.
`obsdim = 1` means the matrix `X` (and `Y`) has size #samples x #dimension
`obsdim = 2` means the matrix `X` (and `Y`) has size #dimension x #samples
"""
kernelmatrix

function kernelmatrix(
    κ::Kernel,
    X::AbstractVector{<:Real};
    obsdim::Int = defaultobs,
)
    kernelmatrix(κ, reshape(X, 1, :), obsdim = 2)
end

function kernelmatrix(κ::Kernel, X::AbstractVector)
    kernelmatrix(κ, X, X) #TODO Can be optimized later
end

function kernelmatrix(κ::Kernel, X::AbstractVector, Y::AbstractVector)
    κ.(X, Y')
end

function kernelmatrix(κ::SimpleKernel, X::AbstractMatrix; obsdim::Int = defaultobs)
    @assert obsdim ∈ [1, 2] "obsdim should be 1 or 2 (see docs of `kernelmatrix`))"
    K = map(x -> kappa(κ, x), pairwise(metric(κ), X, dims = obsdim))
end

function kernelmatrix(κ::Kernel, X::AbstractMatrix; obsdim::Int = defaultobs)
    @assert obsdim ∈ [1, 2] "obsdim should be 1 or 2 (see docs of `kernelmatrix`))"
    if obsdim == 1
        kernelmatrix(κ, ColVecs(X'))
    else
        kernelmatrix(κ, ColVecs(X))
    end
end

function kernelmatrix(
    κ::Kernel,
    X::AbstractVector{<:Real},
    Y::AbstractMatrix{<:Real}
    )
    kernelmatrix(κ, reshape(X, 1, :), reshape(Y, 1, :), obsdim = 2)
end

function kernelmatrix(
    κ::SimpleKernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim = defaultobs,
)
    @assert obsdim ∈ [1, 2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    if !check_dims(X, Y, feature_dim(obsdim), obsdim)
        throw(DimensionMismatch("X $(size(X)) and Y $(size(Y)) do not have the same number of features on the dimension : $(feature_dim(obsdim))"))
    end
    _kernelmatrix(κ, X, Y, obsdim)
end

@inline _kernelmatrix(κ::SimpleKernel, X, Y, obsdim) =
    map(x -> kappa(κ, x), pairwise(metric(κ), X, Y, dims = obsdim))

"""
    kerneldiagmatrix(κ::Kernel, X; obsdim::Int = 2)

Calculate the diagonal matrix of `X` with respect to kernel `κ`
`obsdim = 1` means the matrix `X` has size #samples x #dimension
`obsdim = 2` means the matrix `X` has size #dimension x #samples
"""
kerneldiagmatrix

function kerneldiagmatrix(
    κ::Kernel,
    X::AbstractMatrix;
    obsdim::Int = defaultobs
    )
    @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    if obsdim == 1
        kerneldiagmatrix(κ, ColVecs(X'))
    else
        kerneldiagmatrix(κ, ColVecs(X))
    end
end

function kerneldiagmatrix(κ::Kernel, X::AbstractVector)
    κ.(X, X)
end

"""
    kerneldiagmatrix!(K::AbstractVector, κ::Kernel, X; obsdim::Int = 2)

In place version of [`kerneldiagmatrix`](@ref)
"""
function kerneldiagmatrix!(
    K::AbstractVector,
    κ::Kernel,
    X::AbstractMatrix;
    obsdim::Int = defaultobs
    )
    @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    if length(K) != size(X,obsdim)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
    end
    if obsdim == 1
        kerneldiagmatrix!(K, κ, ColVecs(X'))
    else
        kerneldiagmatrix!(K, κ, ColVecs(X))
    end
    return K
end

function kerneldiagmatrix!(
    K::AbstractVector,
    κ::Kernel,
    X::AbstractVector
    )
    map!(κ, K, X, X)
end
