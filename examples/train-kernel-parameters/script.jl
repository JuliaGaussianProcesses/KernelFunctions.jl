# # Kernel Ridge Regression

# In this example we show the two main methods to perform regression on a kernel from KernelFunctions.jl.

# ## We load KernelFunctions and some other packages

using KernelFunctions
using LinearAlgebra
using Distributions
using Plots;
default(; lw=2.0, legendfontsize=15.0);
using BenchmarkTools
using Flux
using Flux: Optimise
using Zygote
using Random: seed!
seed!(42)
using ParameterHandling

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

# ## Method 1
# The first method is to rebuild the parametrized kernel from a vector of parameters 
# in each evaluation of the cost fuction. This is similar to the approach taken in 
# [Stheno.jl](https://github.com/JuliaGaussianProcesses/Stheno.jl).


# ### Base Approach
# A simple way to ensure that the kernel parameters are positive
# is to optimize over the logarithm of the parameters. 

# To train the kernel parameters via ForwardDiff.jl
# we need to create a function creating a kernel from an array.

function kernelcall(θ)
    return (exp(θ[1]) * SqExponentialKernel() + exp(θ[2]) * Matern32Kernel()) ∘
           ScaleTransform(exp(θ[3]))
end

# From theory we know the prediction for a test set x given
# the kernel parameters and normalization constant

function f(x, x_train, y_train, θ)
    k = kernelcall(θ[1:3])
    return kernelmatrix(k, x, x_train) *
           ((kernelmatrix(k, x_train) + exp(θ[4]) * I) \ y_train)
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

# ## Training the model

θ = log.([1.1, 0.1, 0.01, 0.001]) # Initial vector
opt = Optimise.ADAGrad(0.5)

# The loss with our starting point :

loss(θ)

# Cost for one step

@benchmark let θt = θ[:], optt = Optimise.ADAGrad(0.5)
    grads = only((Zygote.gradient(loss, θt))) # We compute the gradients given the kernel parameters and regularization
    Optimise.update!(optt, θt, grads)
end

# The optimization 

anim = Animation()
for i in 1:25
    grads = only((Zygote.gradient(loss, θ))) # We compute the gradients given the kernel parameters and regularization
    Optimise.update!(opt, θ, grads)
    scatter(
        x_train, y_train; lab="data", title="i = $(i), Loss = $(round(loss(θ), digits = 4))"
    )
    plot!(x_test, sinc; lab="true function")
    plot!(x_test, f(x_test, x_train, y_train, θ); lab="Prediction", lw=3.0)
    frame(anim)
end
gif(anim)

# Final loss
loss(θ)

# ### ParameterHandling.jl
# Alternatively, we can use the [ParameterHandling.jl](https://github.com/invenia/ParameterHandling.jl) package 
# to handle the requirement that all kernel parameters should be positive. 

raw_initial_θ = (
    k1 = positive(1.1),
    k2 = positive(0.1),
    k3 = positive(0.01),
    noise_var=positive(0.001),
)

flat_θ, unflatten = ParameterHandling.value_flatten(raw_initial_θ);

function kernelcall(θ)
    return (θ.k1 * SqExponentialKernel() + θ.k2 * Matern32Kernel()) ∘
           ScaleTransform(θ.k3)
end

function f(x, x_train, y_train, θ)
    k = kernelcall(θ)
    return kernelmatrix(k, x, x_train) *
           ((kernelmatrix(k, x_train) + θ.noise_var * I) \ y_train)
end

function loss(θ)
    ŷ = f(x_train, x_train, y_train, θ)
    return sum(abs2, y_train - ŷ) + θ.noise_var * norm(ŷ)
end

initial_θ = ParameterHandling.value(raw_initial_θ)

# The loss with our starting point :

(loss ∘ unflatten)(flat_θ)

# ## Training the model

# ### Cost per step

@benchmark let θt = flat_θ[:], optt = Optimise.ADAGrad(0.5)
    grads = (Zygote.gradient(loss ∘ unflatten, θt))[1] # We compute the gradients given the kernel parameters and regularization
    Optimise.update!(optt, θt, grads)
end

opt = Optimise.ADAGrad(0.5)
for i in 1:25
    grads = (Zygote.gradient(loss ∘ unflatten, flat_θ))[1] # We compute the gradients given the kernel parameters and regularization
    Optimise.update!(opt, flat_θ, grads)
end

# Final loss

(loss ∘ unflatten)(flat_θ)


# ## Method 2: Functor
# An alternative method is to use tools from Flux.jl.

# raw_initial_θ = (
#     k1 = positive(1.1),
#     k2 = positive(0.1),
#     k3 = positive(0.01),
#     noise_var=positive(0.001),
# )
k1 = [1.1]
k2 = [0.1]
k3 = [0.01]
noise_var = log.([0.001])

kernel = (ScaledKernel(SqExponentialKernel(), relu.(k1)) + ScaledKernel(Matern32Kernel(), k2)) ∘
    ScaleTransform(map(exp,k3))

θ = Flux.params(k1, k2, k3, noise_var)

# kernel = (ScaledKernel(SqExponentialKernel(), softplus(θ[1])) + ScaledKernel(Matern32Kernel(), θ[2])) ∘
#     ScaleTransform(θ[3])

# This next 

# function loss2()
#     ŷ = kernelmatrix(kernel, x_train, x_train) * ((kernelmatrix(kernel, x_train) + θ[4][1] * I) \ y_train)
#     return sum(abs2, y_train - ŷ) + θ[4][1] * norm(ŷ)
# end

function loss()
    ŷ = kernelmatrix(kernel, x_train, x_train) * ((kernelmatrix(kernel, x_train)) \ y_train)
    return sum(abs2, y_train - ŷ) + only(exp.(noise_var) .* norm(ŷ))
end

function f(x, x_train, y_train)
    return kernelmatrix(kernel, x, x_train) *
           ((kernelmatrix(kernel, x_train) + only(exp.(noise_var)) * I) \ y_train)
end


grads = Flux.gradient(loss, θ)
for p in θ
    println(grads[p])
end


grads = Flux.gradient(loss, θ)

η = 0.1 # Learning Rate
opt = Optimise.ADAGrad(η)
# for p in θ
#   update!(p, η * grads[p])
# end

anim = Animation()
for i in 1:25
    Optimise.update!(opt, θ, grads)
    println(θ)

    scatter(
        x_train, y_train; lab="data", title="i = $(i), Loss = $(round(loss(), digits = 4))"
    )
    plot!(x_test, sinc; lab="true function")
    plot!(x_test, f(x_test, x_train, y_train); lab="Prediction", lw=3.0)
    frame(anim)
end

gif(anim)