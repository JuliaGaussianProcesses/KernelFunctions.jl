"""
    WienerKernel{i}()

i-times integrated Wiener process kernel function given by
```julia
    κ(x, y) =  kᵢ(x, y)
```

For i=-1, this is just the white noise covariance, see WhiteKernel.\\
For i= 0, this is the Wiener process covariance,\\
for i= 1, this is the integrated Wiener process covariance (velocity),\\
for i= 2, this is the twice-integrated Wiener process covariance (accel.),\\
for i= 3, this is the thrice-integrated Wiener process covariance. where `kᵢ` is given by\\

```julia
    k₋₁(x, y) =  δ(x, y)
    i >= 0, kᵢ(x, y) = 1 / ai * min(x, y)^(2i + 1) + bi * min(x, y)^(i + 1) * |x - y| * ri(x, y),
    with the coefficients ai, bi and the residual ri(x, y) defined as follows:
        i = 0, ai =   1, bi = 0
        i = 1, ai =   3, bi = 1/  2, ri(x, y) = 1
        i = 2, ai =  20, bi = 1/ 12, ri(x, y) = x + y - 1 / 2 * min(x, y)
        i = 3, ai = 252, bi = 1/720, ri(x, y) = 5 * max(x, y)² + 2 * x * z + 3 * min(x, y)²
```

**References:**\\
See the paper *Probabilistic ODE Solvers with Runge-Kutta Means* by Schober, Duvenaud and Hennig, NIPS, 2014, for more details.

"""
struct WienerKernel{I} <: BaseKernel
    function WienerKernel{I}() where I
        @assert I ∈ (-1, 0, 1, 2, 3) "Invalid parameter i=$(I). Should be -1, 0, 1, 2 or 3."
        if I == -1
            return WhiteKernel()
        end
        return new{I}()
    end
end

function WienerKernel(;i::Integer=0)
    return WienerKernel{i}()
end

function _wiener(κ::WienerKernel{0}, x, y)
    X = sqrt(sum(abs2.(x)))
    Y = sqrt(sum(abs2.(y)))
    return min(X, Y)
end

function _wiener(κ::WienerKernel{1}, x, y)
    X = sqrt(sum(abs2.(x)))
    Y = sqrt(sum(abs2.(y)))
    minXY = min(X, Y)
    return 1 / 3 * minXY^3 + 1 / 2 * minXY^2 * euclidean(x, y)
end

function _wiener(κ::WienerKernel{2}, x, y)
    X = sqrt(sum(abs2.(x)))
    Y = sqrt(sum(abs2.(y)))
    minXY = min(X, Y)
    return 1 / 20 * minXY^5 + 1 / 12 * minXY^3 * euclidean(x, y) *
        ( X + Y - 1 / 2 * minXY )
end

function _wiener(κ::WienerKernel{3}, x, y)
    X = sqrt(sum(abs2.(x)))
    Y = sqrt(sum(abs2.(y)))
    minXY = min(X, Y)
    return 1 / 252 * minXY^7 + 1 / 720 * minXY^4 * euclidean(x, y) *
        ( 5 * max(X, Y)^2 + 2 * X * Y + 3 * minXY^2 )
end

function kappa(κ::WienerKernel, x, y)
    return _wiener(κ, x, y)
end

(κ::WienerKernel)(x::Real, y::Real) = kappa(κ, x, y)

function _kernel(
    κ::WienerKernel,
    x::AbstractVector,
    y::AbstractVector;
    obsdim::Int = defaultobs
)
    @assert length(x) == length(y) "x and y don't have the same dimension!"
    return kappa(κ, x, y)
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::WienerKernel,
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
                @inbounds @views K[i,j] = _kernel(κ, X[i,:], Y[j,:])
            end
        end
    else
        for j = 1:size(K, 2)
            for i = 1:size(K, 1)
                @inbounds @views K[i,j] = _kernel(κ, X[:,i], Y[:,j])
            end
        end
    end
    return K
end

function kernelmatrix(
    κ::WienerKernel,
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
                @inbounds @views K[i,j] = _kernel(κ, X[i,:], Y[j,:])
            end
        end
    else
        for j = 1:size(K, 2)
            for i = 1:size(K, 1)
                @inbounds @views K[i,j] = _kernel(κ, X[:,i], Y[:,j])
            end
        end
    end
    return K
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::WienerKernel,
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
    κ::WienerKernel,
    X::AbstractMatrix;
    obsdim::Int = defaultobs
)
    return kernelmatrix(κ, X, X; obsdim=obsdim)
end

Base.show(io::IO, κ::WienerKernel{I}) where I = print(io, "Wiener Kernel $(I)-times integrated")
