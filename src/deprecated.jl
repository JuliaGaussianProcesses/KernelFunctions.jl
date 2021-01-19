# TODO: remove tests when removed
@deprecate MahalanobisKernel(; P::AbstractMatrix{<:Real}) transform(
    SqExponentialKernel(), LinearTransform(sqrt(2) .* cholesky(P).U)
)

# TODO: remove keyword argument `maha` when removed
@deprecate PiecewisePolynomialKernel{V}(A::AbstractMatrix{<:Real}) where {V} transform(
    PiecewisePolynomialKernel{V}(size(A, 1)), LinearTransform(cholesky(A).U)
)

# TODO: remove tests when removed
@deprecate function kerneldiagmatrix(κ::Kernel, X; obsdim::Int=2)
    return kernelmatrix_diag(κ, X; obsdim=obsdim)
end
@deprecate function kerneldiagmatrix(κ::Kernel, X, Y; obsdim::Int=2)
    return kernelmatrix_diag(κ, X, Y; obsdim=obsdim)
end
@deprecate function kerneldiagmatrix!(K::AbstractVector, κ::Kernel, X; obsdim::Int=2)
    return kernelmatrix_diag!(K, κ, X; obsdim=obsdim)
end
@deprecate function kerneldiagmatrix!(K::AbstractVector, κ::Kernel, X, Y; obsdim::Int=2)
    return kernelmatrix_diag!(K, κ, X, Y; obsdim=obsdim)
end
