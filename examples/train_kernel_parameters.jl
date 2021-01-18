# -*- coding: utf-8 -*-
# # Kernel Ridge Regression

# ## We load KernelFunctions and some other packages

using KernelFunctions
using LinearAlgebra
using Distributions
using Plots;
default(; lw=2.0, legendfontsize=15.0);
using Flux: Optimise
using ForwardDiff
using Random: seed!
seed!(42)

# ## Data Generation
# We generated data in 1 dimension

xmin = -3;
xmax = 3; # Bounds of the data
N = 50 # Number of samples
x_train = rand(Uniform(xmin, xmax), N) # We sample 100 random samples
σ = 0.1
y_train = sinc.(x_train) + randn(N) * σ # We create a function and add some noise
x_test = range(xmin - 0.1, xmax + 0.1; length=300)

# Plot the data

scatter(x_train, y_train; lab="data")
plot!(x_test, sinc; lab="true function")

# ## Kernel training
# To train the kernel parameters via ForwardDiff.jl
# we need to create a function creating a kernel from an array

kernelcall(θ) = transform(
    exp(θ[1]) * SqExponentialKernel(),# + exp(θ[2]) * Matern32Kernel(),
    exp(θ[3]),
)

# From theory we know the prediction for a test set x given
# the kernel parameters and normalization constant

function f(x, x_train, y_train, θ)
    k = kernelcall(θ[1:3])
    return kernelmatrix(k, x, x_train) *
           inv(kernelmatrix(k, x_train) + exp(θ[4]) * I) *
           y_train
end

# We look how the prediction looks like
# with starting parameters [1.0, 1.0, 1.0, 1.0] we get :

ŷ = f(x_test, x_train, y_train, log.(ones(4)))
scatter(x_train, y_train; lab="data")
plot!(x_test, sinc; lab="true function")
plot!(x_test, ŷ; lab="prediction")

# We define the loss based on the L2 norm both
# for the loss and the regularization

function loss(θ)
    ŷ = f(x_train, x_train, y_train, θ)
    return sum(abs2, y_train - ŷ) + exp(θ[4]) * norm(ŷ)
end

# The loss with our starting point :

loss(log.(ones(4)))

# ## Training the model

θ = vcat(log.([1.0, 0.0, 0.01]), log(0.001)) # Initial vector
anim = Animation()
opt = Optimise.ADAGrad(0.5)
for i in 1:30
    grads = ForwardDiff.gradient(x -> loss(x), θ) # We compute the gradients given the kernel parameters and regularization
    Δ = Optimise.apply!(opt, θ, grads)
    θ .-= Δ # We apply a simple Gradient descent algorithm
    p = scatter(
        x_train, y_train; lab="data", title="i = $(i), Loss = $(round(loss(θ), digits = 4))"
    )
    plot!(x_test, sinc; lab="true function")
    plot!(x_test, f(x_test, x_train, y_train, θ); lab="Prediction", lw=3.0)
    frame(anim)
end
gif(anim)
