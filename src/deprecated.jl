# TODO: remove tests when removed
@deprecate MahalanobisKernel(; P::AbstractMatrix{<:Real}) transform(SqExponentialKernel(), LinearTransform(cholesky(P).U))

# TODO: remove keyword argument `maha` when removed
@deprecate PiecewisePolynomialKernel{V}(A::AbstractMatrix{<:Real}) where V transform(PiecewisePolynomialKernel{V}(size(A, 1)), LinearTransform(cholesky(A).U))
