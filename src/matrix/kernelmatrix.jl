"""
    kernelmatrix!(K::AbstractMatrix, κ::Kernel, X; obsdim::Integer = 2)
    kernelmatrix!(K::AbstractMatrix, κ::Kernel, X, Y; obsdim::Integer = 2)

In-place version of [`kernelmatrix`](@ref) where pre-allocated matrix `K` will be overwritten with the kernel matrix.
"""
kernelmatrix!

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
    kernelmatrix!(K, κ, vec_of_vecs(X, obsdim = obsdim))
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::Kernel,
    X::AbstractVector
    )
    if !check_dims(K, X, X)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
    end
    K .= κ.(X, X')
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
    kernelmatrix!(K, κ, vec_of_vecs(X, obsdim = obsdim), vec_of_vecs(Y, obsdim = obsdim))

end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::Kernel,
    X::AbstractVector,
    Y::AbstractVector
    )
    if !check_dims(K, X, Y)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X)) and Y $(size(Y))"))
    end
    K .= κ.(X, Y')
end

"""
    kernelmatrix(κ::Kernel, X; obsdim::Int = 2)
    kernelmatrix(κ::Kernel, X, Y; obsdim::Int = 2)

Calculate the kernel matrix of `X` (and `Y`) with respect to kernel `κ`.
`obsdim = 1` means the matrix `X` (and `Y`) has size #samples x #dimension
`obsdim = 2` means the matrix `X` (and `Y`) has size #dimension x #samples
"""
kernelmatrix

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
    kernelmatrix(κ, vec_of_vecs(X, obsdim = obsdim))
end

function kernelmatrix(
    κ::SimpleKernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim = defaultobs,
)
    @assert obsdim ∈ [1, 2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    if !check_dims(X, Y, feature_dim(obsdim))
        throw(DimensionMismatch("X $(size(X)) and Y $(size(Y)) do not have the same number of features on the dimension : $(feature_dim(obsdim))"))
    end
    map(x -> kappa(κ, x), pairwise(metric(κ), X, Y, dims = obsdim))
end

function kernelmatrix(κ::Kernel, X::AbstractMatrix, Y::AbstractMatrix; obsdim::Int = defaultobs)
    kernelmatrix(κ, vec_of_vecs(X, obsdim = obsdim), vec_of_vecs(Y, obsdim = obsdim))
end

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
    kerneldiagmatrix(κ, vec_of_vecs(X, obsdim = obsdim))
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
    if length(K) != size(X,obsdim)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
    end
    kerneldiagmatrix!(K, κ, vec_of_vecs(X, obsdim = obsdim))
    return K
end

function kerneldiagmatrix!(
    K::AbstractVector,
    κ::Kernel,
    X::AbstractVector
    )
    if length(K) != length(X)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(length(X))"))
    end
    map!(κ, K, X, X)
end
