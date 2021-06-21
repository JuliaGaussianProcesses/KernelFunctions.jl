# # Support Vector Machine
using KernelFunctions
using Distributions, LinearAlgebra
using Plots

N = 100  # Number of samples
μ = randn(2, 2)  # Random Centers
xgrid = range(-3, 3; length=100)  # Create a grid
Xgrid = hcat(collect.(Iterators.product(xgrid, xgrid))...)'  # Combine into a 2D grid
y = rand([-1, 1], N)  # Select randomly between the two classes
X = zeros(N, 2)
X[y .== 1, :] = rand(MvNormal(μ[:, 1], I), count(y .== 1))'  # Attribute samples from class 1
X[y .== -1, :] = rand(MvNormal(μ[:, 2], I), count(y .== -1))'  # Attribute samples from class 2

k = compose(SqExponentialKernel(), ScaleTransform(2.0))  # Create kernel function
function f(x, k, λ)
    return kernelmatrix(k, x, X; obsdim=1) *
           inv(kernelmatrix(k, X; obsdim=1) + exp(λ[1]) * I) *
           y
end  # Optimal prediction f
svmloss(y, ŷ, λ) = ŷ -> sum(maximum.(0.0, 1 - y * ŷ)) - λ * norm(ŷ)(f(X, k, λ))  # Total svm loss with regularisation
pred = f(Xgrid, k, 0.1)  # Compute prediction on a grid
contourf(xgrid, xgrid, pred)
scatter!(eachcol(X)...; color=y, lab="data")
