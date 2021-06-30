# # Kernel Ridge Regression
#
# !!! warning
#     This example is under construction

# Setup

using KernelFunctions
using MLDataUtils
using Zygote
using Flux
using Distributions, LinearAlgebra
using Plots

Flux.@functor SqExponentialKernel
Flux.@functor ScaleTransform
Flux.@functor KernelSum
Flux.@functor Matern32Kernel

# Generate date

xmin = -3;
xmax = 3;
x = range(xmin, xmax; length=100)
x_test = range(xmin, xmax; length=300)
x, y = noisy_function(sinc, x; noise=0.1)
X = RowVecs(reshape(x, :, 1))
X_test = RowVecs(reshape(x_test, :, 1))
#md nothing #hide

# Set up kernel and regularisation parameter

k = SqExponentialKernel() + Matern32Kernel() ∘ ScaleTransform(2.0)
λ = [-1.0]
#md nothing #hide

#

f(x, k, λ) = kernelmatrix(k, x, X) / (kernelmatrix(k, X) + exp(λ[1]) * I) * y
f(X, k, 1.0)

#

loss(k, λ) = (ŷ -> sum(y - ŷ) / length(y) + exp(λ[1]) * norm(ŷ))(f(X, k, λ))
loss(k, λ)

#

ps = Flux.params(k)
push!(ps, λ)
opt = Flux.Momentum(0.1)
#md nothing #hide

plots = []
for i in 1:10
    grads = Zygote.gradient(() -> loss(k, λ), ps)
    Flux.Optimise.update!(opt, ps, grads)
    p = Plots.scatter(x, y; lab="data", title="Loss = $(loss(k,λ))")
    Plots.plot!(x_test, f(X_test, k, λ); lab="Prediction", lw=3.0)
    push!(plots, p)
end

#

l = @layout grid(10, 1)
plot(plots...; layout=l, size=(300, 1500))
