"""
    PiecewisePolynomialKernel{V}(maha::AbstractMatrix)

Piecewise Polynomial covariance function with compact support, V = 0,1,2,3.
The kernel functions are 2v times continuously differentiable and the corresponding
processes are hence v times  mean-square differentiable. The kernel function is:
```math
    κ(x, y) = max(1 - r, 0)^(j + V) * f(r, j) with j = floor(D / 2) + V + 1
```
where `r` is the Mahalanobis distance mahalanobis(x,y) with `maha` as the metric.

"""
struct PiecewisePolynomialKernel{V, A<:AbstractMatrix{<:Real}} <: BaseKernel
    maha::A
    function PiecewisePolynomialKernel{V}(maha::AbstractMatrix{<:Real}) where V
        V in (0, 1, 2, 3) || error("Invalid paramter v=$(V). Should be 0, 1, 2 or 3.")
        LinearAlgebra.checksquare(maha)
        return new{V,typeof(maha)}(maha)
    end
end

function PiecewisePolynomialKernel(;v::Integer=0, maha::AbstractMatrix{<:Real})
    return PiecewisePolynomialKernel{v}(maha)
end

_f(κ::PiecewisePolynomialKernel{0}, r, j) = 1
_f(κ::PiecewisePolynomialKernel{1}, r, j) = 1 + (j + 1) * r
_f(κ::PiecewisePolynomialKernel{2}, r, j) = 1 + (j + 2) * r + (j^2 + 4 * j + 3) / 3 * r.^2
_f(κ::PiecewisePolynomialKernel{3}, r, j) = 1 + (j + 3) * r +
    (6 * j^2 + 36j + 45) / 15 * r.^2 + (j^3 + 9 * j^2 + 23j + 15) / 15 * r.^3

function _piecewisepolynomial(κ::PiecewisePolynomialKernel{V}, r, j) where V
    return max(1 - r, 0)^(j + V) * _f(κ, r, j)
end

function kappa(
    κ::PiecewisePolynomialKernel{V},
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
) where {V}
    r = evaluate(metric(κ), x, y)
    j = div(size(x, 2), 1) + V + 1
    return _piecewisepolynomial(κ, r, j)
end

function _kernel(
    κ::PiecewisePolynomialKernel,
    x::AbstractVector,
    y::AbstractVector;
    obsdim::Int = defaultobs,
)
    @assert length(x) == length(y) "x and y don't have the same dimension!"
    return kappa(κ,x,y)
end

function kernelmatrix(
    κ::PiecewisePolynomialKernel{V},
    X::AbstractMatrix;
    obsdim::Int = defaultobs
) where {V}
    j = div(size(X, feature_dim(obsdim)), 2) + V + 1
    return map(r->_piecewisepolynomial(κ, r, j), pairwise(metric(κ), X; dims=obsdim))
end

function _kernelmatrix(κ::PiecewisePolynomialKernel{V}, X, Y, obsdim) where {V}
    j = div(size(X, feature_dim(obsdim)), 2) + V + 1
    return map(r->_piecewisepolynomial(κ, r, j), pairwise(metric(κ), X, Y; dims=obsdim))
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::PiecewisePolynomialKernel{V},
    X::AbstractMatrix;
    obsdim::Int = defaultobs
) where {V}
    @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    if !check_dims(K, X, X, feature_dim(obsdim), obsdim)
        throw(DimensionMismatch(
            "Dimensions of the target array K $(size(K)) are not consistent with X " *
            "$(size(X))",
        ))
    end
    j = div(size(X, feature_dim(obsdim)), 2) + V + 1
    return map!(r->_piecewisepolynomial(κ,r,j), K, pairwise(metric(κ), X; dims=obsdim))
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::PiecewisePolynomialKernel{V},
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs,
) where {V}
    @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    if !check_dims(K, X, Y, feature_dim(obsdim), obsdim)
        throw(DimensionMismatch(
            "Dimensions $(size(K)) of the target array K are not consistent with X " *
            "($(size(X))) and Y ($(size(Y)))",
        ))
    end
    j = div(size(X, feature_dim(obsdim)), 2) + V + 1
    return map!(r->_piecewisepolynomial(κ,r,j), K, pairwise(metric(κ), X, Y; dims=obsdim))
end

metric(κ::PiecewisePolynomialKernel) = Mahalanobis(κ.maha)

function Base.show(io::IO, κ::PiecewisePolynomialKernel{V}) where {V}
    print(io, "Piecewise Polynomial Kernel (v = ", V, ", size(maha) = ", size(κ.maha), ")")
end
