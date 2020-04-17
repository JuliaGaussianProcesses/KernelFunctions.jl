"""
    NeuralNetworkKernel()

Neural network kernel function with a single parameter for the distance
measure. The kernel function is parameterized as:

```julia
    Îº(x, y) =  asin(x' * y / sqrt[(1 + x' * x) * (1 + y' * y)])
```

"""
struct NeuralNetworkKernel <: BaseKernel end

function kappa(κ::NeuralNetworkKernel, x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    return asin(dot(x, y) / sqrt((1 + sum(abs2, x)) * (1 + sum(abs2, y))))
end

function _kernel(
        κ::Kernel,
        x::AbstractVector,
        y::AbstractVector;
        obsdim::Int = defaultobs
    )
    @assert length(x) == length(y) "x and y don't have the same dimension!"
    kappa(κ, x, y)
end

(κ::NeuralNetworkKernel)(x::Real, y::Real) = asin(x * y/sqrt((1 + x^2) * (1 + y^2)))

function kernelmatrix!(
    K::AbstractMatrix,
    κ::NeuralNetworkKernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs
)
    @assert obsdim ∈ [1, 2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    if !check_dims(K, X, X, feature_dim(obsdim), obsdim)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
    end

    if obsdim == 1
        for j = 1:size(K, 2)
            for i = 1:size(K, 1)
                @inbounds @views K[i,j] = kappa(κ, X[i,:], Y[j,:])
            end
        end
    else
        for j = 1:size(K, 2)
            for i = 1:size(K, 1)
                @inbounds @views K[i,j] = kappa(κ, X[:,i], Y[:,j])
            end
        end
    end
    return K
end

function kernelmatrix(
    κ::NeuralNetworkKernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs
)
    @assert obsdim ∈ [1, 2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    if !check_dims(X, Y, feature_dim(obsdim), obsdim)
        throw(DimensionMismatch("X $(size(X)) and Y $(size(Y)) do not have the same number of features on the dimension : $(feature_dim(obsdim))"))
    end
    if obsdim == 1
        outdim = size(X, 1)
    else
        outdim = size(X, 2)
    end
    K = zeros(outdim, outdim)
    if obsdim == 1
        for j = 1:size(K, 2)
            for i = 1:size(K, 1)
                @inbounds @views K[i,j] = kappa(κ, X[i,:], Y[j,:])
            end
        end
    else
        for j = 1:size(K, 2)
            for i = 1:size(K, 1)
                @inbounds @views K[i,j] = kappa(κ, X[:,i], Y[:,j])
            end
        end
    end
    return K
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::NeuralNetworkKernel,
    X::AbstractMatrix;
    obsdim::Int = defaultobs
)
    @assert obsdim ∈ [1, 2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    if !check_dims(K, X, X, feature_dim(obsdim), obsdim)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
    end
    kernelmatrix!(K, κ, X, X; obsdim=obsdim)
    return K
end

function kernelmatrix(
    κ::NeuralNetworkKernel,
    X::AbstractMatrix;
    obsdim::Int = defaultobs
)
    return kernelmatrix(κ, X, X; obsdim=obsdim)
end

Base.show(io::IO, κ::NeuralNetworkKernel) = print(io, "Neural Network Kernel")
