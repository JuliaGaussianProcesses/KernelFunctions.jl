# # Support Vector Machines
# ## Package loading
using KernelFunctions
using Distributions, LinearAlgebra
using Plots; default(legendfontsize = 15.0, ms = 5.0)

# ## Data Generation
# ### We first generate a mixture of two Gaussians in 2 dimensions
xmin = -3; xmax = 3 # Limits for sampling μ₁ and μ₂
μ = rand(Uniform(xmin, xmax), 2, 2) # Sample 2 Random Centers
# ### We then sample both y and x
N = 100 # Number of samples
y = rand((-1, 1), N) # Select randomly between the two classes
x = Vector{Vector{Float64}}(undef, N) # We preallocate x
x[y .== 1] = [rand(MvNormal(μ[:, 1], I)) for _ in 1:count(y.==1)] # Features for samples of class 1
x[y .== -1] = [rand(MvNormal(μ[:, 2], I)) for _ in 1:count(y.==-1)] # Features for samples of class 2
scatter(getindex.(x[y .== 1], 1), getindex.(x[y .== 1], 2), label = "y = 1", title = "Data")
scatter!(getindex.(x[y .== -1], 1), getindex.(x[y .== -1], 2), label = "y = 2")

# ## Model Definition
# TODO Write theory here
# ### We create a kernel k
k = SqExponentialKernel() # SqExponentialKernel or RBFKernel
λ = 1.0 # Regularization parameter

# ### We create a function to return the optimal prediction for a
# test data `x_new`
function f(x_new, x, y, k, λ)
    kernelmatrix(k, x_new, x) * inv(kernelmatrix(k, x) + λ * I) * y # Optimal prediction f
end

# ### We also compute the total loss of the model that we want to minimize
hingeloss(y, ŷ) = maximum(zero(ŷ), 1 - y * ŷ) # hingeloss function
function reg_hingeloss(k, x, y, λ)
    ŷ = f(x, x, y, k, λ)
    return sum(hingeloss.(y, ŷ)) - λ * norm(ŷ) # Total svm loss with regularisation
end
# ### We create a 2D grid based on the maximum values of the data
N_test = 200 # Size of the grid
xgrid = range(extrema(vcat(x...)).*1.1..., length=N_test) # Create a 1D grid
xgrid = vec(collect.(Iterators.product(xgrid, xgrid))) #Combine into a 2D grid
# ### We predict the value of y on this grid on plot it against the data
y_grid = f(xgrid, x, y, k, λ) #Compute prediction on a grid
contourf(xgrid, xgrid, reshape(y_grid, N_test, N_test)', label =  "Predictions", title="Trained model")
scatter!(getindex.(x[y .== 1], 1), getindex.(x[y .== 1], 2), label = "y = 1")
scatter!(getindex.(x[y .== -1], 1), getindex.(x[y .== -1], 2), label = "y = 2")
