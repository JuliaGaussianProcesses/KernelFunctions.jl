# # Support Vector Machine
#
# In this notebook we show how you can use KernelFunctions.jl to generate
# kernel matrices for classification with a support vector machine, as
# implemented by LIBSVM.

using Distributions
using KernelFunctions
using LIBSVM
using LinearAlgebra
using Plots
using Random

## Set seed
Random.seed!(1234);

# ## Generate half-moon dataset

# Number of samples per class:
n1 = n2 = 50;

# We generate data based on SciKit-Learn's sklearn.datasets.make_moons function:

angle1 = range(0, π; length=n1)
angle2 = range(0, π; length=n2)
X1 = [cos.(angle1) sin.(angle1)] .+ 0.1randn(n1, 2)
X2 = [1 .- cos.(angle2) 1 .- sin.(angle2) .- 0.5] .+ 0.1randn(n2, 2)
X = [X1; X2]
x_train = RowVecs(X)
y_train = vcat(fill(-1, nout), fill(1, nin));

# Create a 100×100 2D grid for evaluation:
test_range = range(floor(Int, minimum(X)), ceil(Int, maximum(X)); length=100)
x_test = ColVecs(mapreduce(collect, hcat, Iterators.product(test_range, test_range)));

# ## SVM model
#
# Create kernel function:
k = SqExponentialKernel() ∘ ScaleTransform(1.5)

# [LIBSVM](https://github.com/JuliaML/LIBSVM.jl) can make use of a pre-computed kernel matrix.
# KernelFunctions.jl can be used to produce that.
#
# Precomputed matrix for training
model = svmtrain(kernelmatrix(k, x_train), y_train; kernel=LIBSVM.Kernel.Precomputed)

# Precomputed matrix for prediction
y_pred, _ = svmpredict(model, kernelmatrix(k, x_train, x_test));

# Visualize prediction on a grid:
plot(; xlim=extrema(test_range), ylim=extrema(test_range), aspect_ratio=1)
contourf!(test_range, test_range, y_pred; levels=1, color=cgrad(:redsblues), alpha=0.7)
scatter!(X1[:, 1], X1[:, 2]; color=:red, label="class 1")
scatter!(X2[:, 1], X2[:, 2]; color=:blue, label="class 2")
