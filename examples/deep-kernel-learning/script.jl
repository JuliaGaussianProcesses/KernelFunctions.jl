using KernelFunctions
using MLDataUtils
using Zygote
using Flux
using Distributions, LinearAlgebra
using Plots

Flux.@functor SqExponentialKernel
Flux.@functor KernelSum
Flux.@functor Matern32Kernel
Flux.@functor FunctionTransform

neuralnet = Chain(Dense(1, 3), Dense(3, 2))
k = SqExponentialKernel() ∘ FunctionTransform(neuralnet)
xmin = -3;
xmax = 3;
x = range(xmin, xmax; length=100)
x_test = rand(Uniform(xmin, xmax), 200)
x, y = noisy_function(sinc, x; noise=0.1)
X = reshape(x, :, 1)
λ = [0.1]
function f(x, k, λ)
    return kernelmatrix(k, X, x; obsdim=1) *
           inv(kernelmatrix(k, X; obsdim=1) + exp(λ[1]) * I) *
           y
end
f(X, k, 1.0)
loss(k, λ) = ŷ -> sum(y - ŷ) / length(y) + exp(λ[1]) * norm(ŷ)(f(X, k, λ))
loss(k, λ)
ps = Flux.params(k)
# push!(ps,λ)
opt = Flux.Momentum(1.0)
##
for i in 1:10
    grads = Zygote.gradient(() -> loss(k, λ), ps)
    Flux.Optimise.update!(opt, ps, grads)
    p = Plots.scatter(x, y; lab="data", title="Loss = $(loss(k,λ))")
    Plots.plot!(x, f(X, k, λ); lab="Prediction", lw=3.0)
    display(p)
end
