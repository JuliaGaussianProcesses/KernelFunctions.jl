# Kernel Ridge Regression
using KernelFunctions
using LinearAlgebra
using Distributions
using Plots
using Flux: Optimise
using ForwardDiff

# We generate our data :
xmin = -3; xmax = 3 # Bounds of the data
N = 50 # Number of samples
σ = 0.1

x_train = rand(Uniform(xmin, xmax), N) # We sample 100 random samples
x_test = range(xmin, xmax, length = 300) # We use x_test to show the prediction function
y = sinc.(x_train) + randn(N) * σ # We create a function and add some noise

# Plot the data
scatter(x_train, y, lab = "data")
plot!(x_test, sinc, lab = "true function")

# Create a function taking kernel parameters and creating a kernel
kernelcall(θ) = transform(
    exp(θ[1]) * SqExponentialKernel(),# + exp(θ[2]) * Matern32Kernel(),
    exp(θ[3]),
)

# Return the prediction given the normalization value λ
function f(x, θ)
    k = kernelcall(θ[1:3])
    kernelmatrix(k, x, x_train) *
    inv(kernelmatrix(k, x_train) + exp(θ[4]) * I) * y
end

# Starting with parameters [1.0, 1.0, 1.0, 1.0] we get :
ŷ = f(x_test, log.(ones(4)))
scatter(x_train, y, lab = "data")
plot!(x_test, sinc, lab = "true function")
plot!(x_test, ŷ, lab = "prediction")

# Create a loss on the training data
function loss(θ)
    ŷ = f(x_train, θ)
    sum(abs2, y - ŷ) + exp(θ[4]) * norm(ŷ)
end

# The loss with our starting point :
loss(log.(ones(4)))

## Training model

θ = vcat(log.([1.0, 0.0, 0.01]), log(0.001))
anim = Animation()
opt = ADAGrad(0.5)
@progress for i = 1:30
    grads = ForwardDiff.gradient(x -> loss(x), θ) # We compute the gradients given the kernel parameters and regularization
    Δ = Optimise.apply!(opt, θ, grads)
    θ .-= Δ # We apply a simple Gradient descent algorithm
    p = scatter(x_train, y, lab = "data", title = "i = $(i), Loss = $(round(loss(θ), digits = 4))")
    plot!(x_test, sinc, lab = "true function")
    plot!(x_test, f(x_test, θ), lab = "Prediction", lw = 3.0)
    frame(anim)
end
gif(anim)
