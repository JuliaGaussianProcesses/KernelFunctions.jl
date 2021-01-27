# TODO: remove tests when removed
@deprecate MahalanobisKernel(; P::AbstractMatrix{<:Real}) transform(
    SqExponentialKernel(), LinearTransform(sqrt(2) .* cholesky(P).U)
)

# TODO: remove keyword argument `maha` when removed
@deprecate PiecewisePolynomialKernel{V}(A::AbstractMatrix{<:Real}) where {V} transform(
    PiecewisePolynomialKernel{V}(size(A, 1)), LinearTransform(cholesky(A).U)
)

Base.@deprecate_binding TensorProduct KernelTensorProduct

# TODO: remove tests when removed
@deprecate kerneldiagmatrix(k::Kernel, x::AbstractVector) kernelmatrix_diag(k, x)
@deprecate kerneldiagmatrix(k::Kernel, x::AbstractVector, y::AbstractVector) kernelmatrix_diag(
    k, x, y
)
@deprecate kerneldiagmatrix!(K, k::Kernel, x::AbstractVector) kernelmatrix_diag!(K, k, x)
@deprecate kerneldiagmatrix!(K, k::Kernel, x::AbstractVector, y::AbstractVector) kernelmatrix_diag!(
    K, k, x, y
)
@deprecate kerneldiagmatrix(k::Kernel, X::AbstractMatrix; obsdim::Int=defaultobs) kernelmatrix_diag(
    k, X; obsdim=obsdim
)
@deprecate kerneldiagmatrix(
    k::Kernel, X::AbstractMatrix, Y::AbstractMatrix; obsdim::Int=defaultobs
) kernelmatrix_diag(k, X, Y; obsdim=obsdim)
@deprecate kerneldiagmatrix!(K, k::Kernel, X::AbstractMatrix; obsdim::Int=defaultobs) kernelmatrix_diag!(
    K, k, X; obsdim=obsdim
)
@deprecate kerneldiagmatrix!(
    K, k::Kernel, X::AbstractMatrix, Y::AbstractMatrix; obsdim::Int=defaultobs
) kernelmatrix_diag!(K, k, X, Y; obsdim=obsdim)
