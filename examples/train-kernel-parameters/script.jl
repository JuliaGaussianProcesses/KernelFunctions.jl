# # Train Kernel Parameters

# Here we show a few ways to train (optimize) the kernel (hyper)parameters at the example of kernel-based regression using KernelFunctions.jl. 
# All options are functionally identical, but differ a little in readability, dependencies, and computational cost. 

# We load KernelFunctions and some other packages. Note that while we use `Zygote` for automatic differentiation and `Flux.optimise` for optimization, you should be able to replace them with your favourite autodiff framework or optimizer. 

using KernelFunctions
using LinearAlgebra
using Distributions
using Plots
using BenchmarkTools
using Flux
using Flux: Optimise
using Zygote
using Random: seed!
seed!(42);

# ## Data Generation
# We generate a toy dataset in 1 dimension:

xmin, xmax = -3, 3  # Bounds of the data
N = 50 # Number of samples
x_train = rand(Uniform(xmin, xmax), N)  # sample the inputs
σ = 0.1
y_train = sinc.(x_train) + randn(N) * σ  # evaluate a function and add some noise
x_test = range(xmin - 0.1, xmax + 0.1; length=300)
nothing #hide

# Plot the data

scatter(x_train, y_train; label="data")
plot!(x_test, sinc; label="true function")

# ## Manual Approach
# The first option is to rebuild the parametrized kernel from a vector of parameters 
# in each evaluation of the cost function. This is similar to the approach taken in 
# [Stheno.jl](https://github.com/JuliaGaussianProcesses/Stheno.jl).

# To train the kernel parameters via [Zygote.jl](https://github.com/FluxML/Zygote.jl),
# we need to create a function creating a kernel from an array.
# A simple way to ensure that the kernel parameters are positive
# is to optimize over the logarithm of the parameters. 

function kernelcall(θ)
    return (exp(θ[1]) * SqExponentialKernel() + exp(θ[2]) * Matern32Kernel()) ∘
           ScaleTransform(exp(θ[3]))
end
nothing #hide

# From theory we know the prediction for a test set x given
# the kernel parameters and normalization constant:

function f(x, x_train, y_train, θ)
    k = kernelcall(θ[1:3])
    return kernelmatrix(k, x, x_train) *
           ((kernelmatrix(k, x_train) + exp(θ[4]) * I) \ y_train)
end
nothing #hide

# Let's look at our prediction.
# With starting parameters `p0` (picked so we get the right local 
# minimum for demonstration) we get:

p0 = [1.1, 0.1, 0.01, 0.001]
θ = log.(p0)
ŷ = f(x_test, x_train, y_train, θ)
scatter(x_train, y_train; label="data")
plot!(x_test, sinc; label="true function")
plot!(x_test, ŷ; label="prediction")

# We define the following loss:

function loss(θ)
    ŷ = f(x_train, x_train, y_train, θ)
    return norm(y_train - ŷ) + exp(θ[4]) * norm(ŷ)
end
nothing #hide

# The loss with our starting point:

loss(θ)

# Computational cost for one step:

@benchmark let
    θ = log.(p0)
    opt = Optimise.ADAGrad(0.5)
    grads = only((Zygote.gradient(loss, θ)))
    Optimise.update!(opt, θ, grads)
end

# ### Training the model

# Setting an initial value and initializing the optimizer: 
θ = log.(p0) # Initial vector
opt = Optimise.ADAGrad(0.5)
nothing #hide

# Optimize

anim = Animation()
for i in 1:15
    grads = only((Zygote.gradient(loss, θ)))
    Optimise.update!(opt, θ, grads)
    scatter(
        x_train, y_train; lab="data", title="i = $(i), Loss = $(round(loss(θ), digits = 4))"
    )
    plot!(x_test, sinc; lab="true function")
    plot!(x_test, f(x_test, x_train, y_train, θ); lab="Prediction", lw=3.0)
    frame(anim)
end
gif(anim, "train-kernel-param.gif"; show_msg=false, fps=15);
nothing; #hide

# ![](train-kernel-param.gif)

# Final loss
loss(θ)

# ## Using ParameterHandling.jl
# Alternatively, we can use the [ParameterHandling.jl](https://github.com/invenia/ParameterHandling.jl) package 
# to handle the requirement that all kernel parameters should be positive. 
# The package also allows arbitrarily nesting named tuples that make the parameters 
# more human readable, without having to remember their position in a flat vector. 

using ParameterHandling

raw_initial_θ = (
    k1=positive(1.1), k2=positive(0.1), k3=positive(0.01), noise_var=positive(0.001)
)

flat_θ, unflatten = ParameterHandling.value_flatten(raw_initial_θ)
flat_θ #hide

# We define a few relevant functions and note that compared to the previous `kernelcall` function, we do not need explicit `exp`s. 

function kernelcall(θ)
    return (θ.k1 * SqExponentialKernel() + θ.k2 * Matern32Kernel()) ∘ ScaleTransform(θ.k3)
end
nothing #hide

function f(x, x_train, y_train, θ)
    k = kernelcall(θ)
    return kernelmatrix(k, x, x_train) *
           ((kernelmatrix(k, x_train) + θ.noise_var * I) \ y_train)
end
nothing #hide

function loss(θ)
    ŷ = f(x_train, x_train, y_train, θ)
    return norm(y_train - ŷ) + θ.noise_var * norm(ŷ)
end
nothing #hide

initial_θ = ParameterHandling.value(raw_initial_θ)
nothing #hide

# The loss at the initial parameter values:

(loss ∘ unflatten)(flat_θ)

# Cost per step

@benchmark let
    θ = flat_θ[:]
    opt = Optimise.ADAGrad(0.5)
    grads = (Zygote.gradient(loss ∘ unflatten, θ))[1]
    Optimise.update!(opt, θ, grads)
end

# ### Training the model

# Optimize

opt = Optimise.ADAGrad(0.5)
for i in 1:15
    grads = (Zygote.gradient(loss ∘ unflatten, flat_θ))[1]
    Optimise.update!(opt, flat_θ, grads)
end
nothing #hide

# Final loss

(loss ∘ unflatten)(flat_θ)

# ## Flux.destructure
# If we don't want to write an explicit function to construct the kernel, we can alternatively use the `Flux.destructure` function. 
# Again, we need to ensure that the parameters are positive. Note that the `exp` function is now part of the loss function, instead of part of the kernel construction. 

# We could also use ParameterHandling.jl here. 
# To do so, one would remove the `exp`s from the loss function below and call `loss ∘ unflatten` as above. 

θ = [1.1, 0.1, 0.01, 0.001]

kernel = (θ[1] * SqExponentialKernel() + θ[2] * Matern32Kernel()) ∘ ScaleTransform(θ[3])

params, kernelc = Flux.destructure(kernel);

# This returns the trainable `params` of the kernel and a function to reconstruct the kernel.
kernelc(params)

# From theory we know the prediction for a test set x given
# the kernel parameters and normalization constant

function f(x, x_train, y_train, θ)
    k = kernelc(θ[1:3])
    return kernelmatrix(k, x, x_train) * ((kernelmatrix(k, x_train) + (θ[4]) * I) \ y_train)
end
nothing #hide


function loss(θ)
    ŷ = f(x_train, x_train, y_train, exp.(θ))
    return norm(y_train - ŷ) + exp(θ[4]) * norm(ŷ)
end
nothing #hide

# Cost for one step

@benchmark let θt = θ[:], optt = Optimise.ADAGrad(0.5)
    grads = only((Zygote.gradient(loss, θt)))
    Optimise.update!(optt, θt, grads)
end

# ### Training the model

# The loss at our initial parameter values:
θ = log.([1.1, 0.1, 0.01, 0.001]) # Initial vector
loss(θ)

# Initialize optimizer 

opt = Optimise.ADAGrad(0.5)
nothing #hide

# Optimize

for i in 1:15
    grads = only((Zygote.gradient(loss, θ)))
    Optimise.update!(opt, θ, grads)
end
nothing #hide

# Final loss
loss(θ)
