using KernelFunctions
using Test

# minimal definition of a custom kernel
struct MyKernel <: Kernel{IdentityTransform} end

KernelFunctions.kappa(::MyKernel, d2::Real) = exp(-d2)
KernelFunctions.metric(::MyKernel) = SqEuclidean()
KernelFunctions.transform(::MyKernel) = IdentityTransform()

@test kappa(MyKernel(), 3) == kappa(SqExponentialKernel(), 3)
@test kappa(MyKernel(), 1, 3) == kappa(SqExponentialKernel(), 1, 3)
@test kappa(MyKernel(), [1, 2], [3, 4]) == kappa(SqExponentialKernel(), [1, 2], [3, 4])
@test kernelmatrix(MyKernel(), [1 2; 3 4], [5 6; 7 8]) == kernelmatrix(SqExponentialKernel(), [1 2; 3 4], [5 6; 7 8])
@test kernelmatrix(MyKernel(), [1 2; 3 4]) == kernelmatrix(SqExponentialKernel(), [1 2; 3 4])

# some syntactic sugar
(κ::MyKernel)(d::Real) = kappa(κ, d)
(κ::MyKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = kappa(κ, x, y)
(κ::MyKernel)(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; obsdim = 2) = kernelmatrix(κ, X, Y; obsdim = obsdim)
(κ::MyKernel)(X::AbstractMatrix{<:Real}; obsdim = 2) = kernelmatrix(κ, X; obsdim = obsdim)

@test MyKernel()(3) == SqExponentialKernel()(3)
@test MyKernel()([1, 2], [3, 4]) == SqExponentialKernel()([1, 2], [3, 4])
@test MyKernel()([1 2; 3 4], [5 6; 7 8]) == SqExponentialKernel()([1 2; 3 4], [5 6; 7 8])
@test MyKernel()([1 2; 3 4]) == SqExponentialKernel()([1 2; 3 4])