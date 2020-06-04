using KernelFunctions
using MLDataUtils
using Zygote
using Flux
using Distributions, LinearAlgebra
using Plots

N = 100 # Number of samples
N_test = 200 # Size of the grid
xmin = -3; xmax = 3
μ = rand(Uniform(xmin, xmax), 2, 2) # Random Centers
xgrid = range(-xmin, xmax, length=N_test) # Create a grid
Xgrid = hcat(collect.(Iterators.product(xgrid, xgrid))...) #Combine into a 2D grid
y = rand((-1, 1), N) # Select randomly between the two classes
X_train = zeros(2, N)
X_train[:, y .== 1] = rand(MvNormal(μ[:, 1], I), count(y.==1)) #Attribute samples from class 1
X_train[:, y .== -1] = rand(MvNormal(μ[:, 2], I), count(y.==-1)) # Attribute samples from class 2
scatter(eachrow(X_train)..., zcolor= y)
## Compute predictions
k = SqExponentialKernel() # Create kernel function
function f(x, k, λ)
    kernelmatrix(k, x, X_train, obsdim=2) * inv(kernelmatrix(k, X_train, obsdim=2) + exp(λ[1]) * I) * y # Optimal prediction f
end
λ = log.([1.0])
function reg_hingeloss(k, λ)
    ŷ = f(X, k, λ)
    return sum(maximum.(0.0, 1 - y * ŷ)) - exp(λ[1]) * norm(ŷ) # Total svm loss with regularisation
end
y_grid = f(Xgrid, k, λ) #Compute prediction on a grid
contourf(xgrid, xgrid, reshape(y_grid, N_test, N_test))
scatter!(eachrow(X_train)..., zcolor=y,lab="data")
