# # Support Vector Machine
#

using Distributions
using KernelFunctions
using LIBSVM
using LinearAlgebra
using Plots
using Random

## Set plotting theme
theme(:wong)

## Set seed
Random.seed!(1234);

# Number of samples:
N = 100;

# Select randomly between two classes:
y_train = rand([-1, 1], N);

# Random attributes for both classes:
X = Matrix{Float64}(undef, 2, N)
rand!(MvNormal(randn(2), I), view(X, :, y_train .== 1))
rand!(MvNormal(randn(2), I), view(X, :, y_train .== -1));
x_train = ColVecs(X);

# Create a 2D grid:
test_range = range(floor(Int, minimum(X)), ceil(Int, maximum(X)); length=100)
x_test = ColVecs(mapreduce(collect, hcat, Iterators.product(test_range, test_range)));

# Create kernel function:
k = SqExponentialKernel() ∘ ScaleTransform(2.0)

# [LIBSVM](https://github.com/JuliaML/LIBSVM.jl) can make use of a pre-computed kernel matrix.
# KernelFunctions.jl can be used to produce that.
# Precomputed matrix for training (corresponds to linear kernel)
model = svmtrain(kernelmatrix(k, x_train), y_train; kernel=LIBSVM.Kernel.Precomputed)

# Precomputed matrix for prediction
y_pr, _ = svmpredict(model, kernelmatrix(k, x_train, x_test));

# Compute prediction on a grid:
contourf(test_range, test_range, y_pr)
scatter!(X[1, :], X[2, :]; color=y_train, lab="data", widen=false)
