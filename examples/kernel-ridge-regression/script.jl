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

xmin = -3;
xmax = 3;
x = range(xmin, xmax; length=100)
x_test = range(xmin, xmax; length=300)
x, y = noisy_function(sinc, x; noise=0.1)
X = reshape(x, :, 1)
X_test = reshape(x_test, :, 1)
k = SqExponentialKernel(1.0)#+Matern32Kernel(2.0)
λ = [-1.0]
function f(x, k, λ)
    return kernelmatrix(k, x, X; obsdim=1) *
           inv(kernelmatrix(k, X; obsdim=1) + exp(λ[1]) * I) *
           y
end
f(X, k, 1.0)
loss(k, λ) = ŷ -> sum(y - ŷ) / length(y) + exp(λ[1]) * norm(ŷ)(f(X, k, λ))
loss(k, λ)
ps = Flux.params(k)
push!(ps, λ)
opt = Flux.Momentum(0.1)
##
for i in 1:10
    grads = Zygote.gradient(() -> loss(k, λ), ps)
    Flux.Optimise.update!(opt, ps, grads)
    p = Plots.scatter(x, y; lab="data", title="Loss = $(loss(k,λ))")
    Plots.plot!(x_test, f(X_test, k, λ); lab="Prediction", lw=3.0)
    display(p)
end
