# # Support Vector Machines

# TODO: introduction
#
# We first load the packages we will need in this notebook:

using Distributions
using KernelFunctions
using LIBSVM
using LinearAlgebra
using Plots
using Random

## Set plotting theme
theme(:wong)
default(; legendfontsize=15.0, ms=5.0);

## Set seed
Random.seed!(1234);

# ## Data Generation
#
# We first generate a mixture of two Gaussians in 2 dimensions

xmin = -3;
xmax = 3; # Limits for sampling μ₁ and μ₂
μ = rand(Uniform(xmin, xmax), 2, 2) # Sample 2 Random Centers

# We then sample both the input $x$ and the class $y$:

N = 100 # Number of samples
y = rand((-1, 1), N) # Select randomly between the two classes
x = Vector{Vector{Float64}}(undef, N) # We preallocate x
x[y .== 1] = [rand(MvNormal(μ[:, 1], I)) for _ in 1:count(y .== 1)] # Features for samples of class 1
x[y .== -1] = [rand(MvNormal(μ[:, 2], I)) for _ in 1:count(y .== -1)] # Features for samples of class 2
scatter(getindex.(x[y .== 1], 1), getindex.(x[y .== 1], 2); label="y = 1", title="Data")
scatter!(getindex.(x[y .== -1], 1), getindex.(x[y .== -1], 2); label="y = 2")

# Select randomly between two classes:
y_train = rand([-1, 1], N);

# Random attributes for both classes:
X = Matrix{Float64}(undef, 2, N)
rand!(MvNormal(randn(2), I), view(X, :, y_train .== 1))
rand!(MvNormal(randn(2), I), view(X, :, y_train .== -1));
x_train = ColVecs(X);

# We create a 2D grid based on the maximum values of the data
test_range = range(floor(Int, minimum(X)), ceil(Int, maximum(X)); length=100)
x_test = ColVecs(mapreduce(collect, hcat, Iterators.product(test_range, test_range)));

N_test = 100 # Size of the grid
xgrid = range(extrema(vcat(x...)) .* 1.1...; length=N_test) # Create a 1D grid
xgrid_v = vec(collect.(Iterators.product(xgrid, xgrid))); # Combine into a 2D grid

# Create kernel function:
k = SqExponentialKernel() ∘ ScaleTransform(2.0)
λ = 1.0; # Regularization parameter

# ### Predictor
# We create a function to return the optimal prediction for a test data `x_new`

# [LIBSVM](https://github.com/JuliaML/LIBSVM.jl) can make use of a pre-computed kernel matrix.
# KernelFunctions.jl can be used to produce that.
# Precomputed matrix for training (corresponds to linear kernel)
model = svmtrain(kernelmatrix(k, x_train), y_train; kernel=LIBSVM.Kernel.Precomputed)

# We predict the value of y on this grid and plot it against the data:

# Precomputed matrix for prediction
y_pr, _ = svmpredict(model, kernelmatrix(k, x_train, x_test));

# Compute prediction on a grid:
contourf(test_range, test_range, y_pr; label="predictions")
scatter!(X[1, :], X[2, :]; color=y_train, lab="data", widen=false)
#scatter!(getindex.(x[y .== 1], 1), getindex.(x[y .== 1], 2); label="y = 1")
#scatter!(getindex.(x[y .== -1], 1), getindex.(x[y .== -1], 2); label="y = 2")
#xlims!(extrema(xgrid))
#ylims!(extrema(xgrid))
