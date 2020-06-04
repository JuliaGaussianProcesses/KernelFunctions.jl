using KernelFunctions
using MLDataUtils
using Zygote
using Flux
using Distributions, LinearAlgebra
using Plots;
pyplot(); default(legendfontsize = 15.0, linewidth = 3.0)
using SliceMap

neuralnet = Chain(Dense(1, 10), Dense(10, 2))

Base.map(
    t::KernelFunctions.FunctionTransform,
    X::AbstractVector;
    obsdim::Int = defaultobs,
) where {T} =
    slicemap(X, dims = 1) do x
        t.f(x)
    end

xmin = -3; xmax = 3

x_train = range(xmin, xmax, length = 100)
x_test = rand(Uniform(xmin, xmax), 200)
x_train, y = noisy_function(x_train; noise = 0.01) do x
    sinc(abs(x) ^ abs(x))
end
X = reshape(x_train, :, 1)
k = transform(SqExponentialKernel(), FunctionTransform(neuralnet))

λ = log.([1.0])
function f(x, k, λ)
    K = kernelmatrix(k, X, x, obsdim = 1)
    return K * inv(K + exp(λ[1]) * I) * y
end
f(X, k, 1.0)
loss(k, λ) = f(X, k, λ) |> ŷ -> sum(abs2, y - ŷ) / length(y) + exp(λ[1]) * norm(ŷ)

@info "Init Loss = $(loss(k, λ))"

ps = Flux.params(k)
push!(ps,λ)

p = Plots.scatter(x_train, y, lab = "data", title = "Loss = $(loss(k, λ))")
Plots.plot!(x_train, f(X, k, λ), lab = "Prediction", lw = 3.0) |> display
##
anim = Animation()
nmax= 10
opt = Flux.ADAGrad(0.001)
@progress for i = 1:nmax
    grads = Zygote.gradient(() -> loss(k, λ), ps)
    Flux.Optimise.update!(opt, ps, grads)
    if i % 100 == 0
        @info "$i/$nmax"
        p = Plots.scatter(x, y, lab = "data", title = "Loss = $(loss(k,λ))")
        Plots.plot!(x, f(X, k, λ), lab = "Prediction", lw = 3.0)
        frame(anim)
    end
end
gif(anim, fps = 5)
