# # Support Vector Machine
#

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
nin = nout = 50;

# Generate data
## based on SciKit-Learn's sklearn.datasets.make_moons function
class1x = cos.(range(0, π; length=nout))
class1y = sin.(range(0, π; length=nout))
class2x = 1 .- cos.(range(0, π; length=nin))
class2y = 1 .- sin.(range(0, π; length=nin)) .- 0.5
X = hcat(vcat(class1x, class2x), vcat(class1y, class2y))
X .+= 0.1randn(size(X))
x_train = RowVecs(X);
y_train = vcat(fill(-1, nout), fill(1, nin));

# Create a 100×100 2D grid for evaluation:
test_range = range(floor(Int, minimum(X)), ceil(Int, maximum(X)); length=100)
x_test = ColVecs(mapreduce(collect, hcat, Iterators.product(test_range, test_range)));

# Create kernel function:
k = SqExponentialKernel() ∘ ScaleTransform(1.5)

# [LIBSVM](https://github.com/JuliaML/LIBSVM.jl) can make use of a pre-computed kernel matrix.
# KernelFunctions.jl can be used to produce that.
#
# Precomputed matrix for training
model = svmtrain(kernelmatrix(k, x_train), y_train; kernel=LIBSVM.Kernel.Precomputed)

# Precomputed matrix for prediction
y_pred, _ = svmpredict(model, kernelmatrix(k, x_train, x_test));

# Compute prediction on a grid:
plot(lim=extrema(test_range))
contourf!(test_range, test_range, y_pred; levels=1, color=cgrad(:redsblues), alpha=0.7)
scatter!(X[y_train.==-1, 1], X[y_train.==-1, 2]; color=:red, label="class 1", widen=false)
scatter!(X[y_train.==+1, 1], X[y_train.==+1, 2]; color=:blue, label="class 2", widen=false)
