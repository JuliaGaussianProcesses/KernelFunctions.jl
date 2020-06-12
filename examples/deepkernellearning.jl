# # Loading dataset
# We use a couple of useful datasets to plot and optimize
# The different hyper-parameters
using KernelFunctions
using MLDataUtils
using Zygote
using Flux
using Distributions, LinearAlgebra
using Plots
using AbstractGPs
pyplot(); default(legendfontsize = 15.0, linewidth = 3.0)
using SliceMap



Base.map(
    t::KernelFunctions.FunctionTransform,
    X::AbstractVector;
    obsdim::Int = defaultobs,
) where {T} =
    slicemap(X, dims = 1) do x
        t.f(x)
    end

# ## Data creation
# We create a simple 1D Problem with very different variations
xmin = -3; xmax = 3 # Limits
noise = 0.01
x_train = rand(Uniform(xmin, xmax), 100) # Training dataset
x_test = range(xmin, xmax, length=200) # Testing dataset
target_f(x) = sinc(abs(x) ^ abs(x)) # We use sinc with a highly varying value
x_train, y = noisy_function(target_f, x_train; noise = 0.01)

# ## Model definition
# We create a neural net with 2 layers and 10 units each
# The data is passed through the NN before being used in the kernel
neuralnet = Chain(Dense(1, 10), Dense(10, 2))
# We use a Squared Exponential Kernel
k = transform(SqExponentialKernel(), FunctionTransform(neuralnet))

f = GP(k)
fx = f(x_train, noise)
fp = posterior(fx, y)
# This compute the log evidence of `y`,
# which is going to be used as the objective
loss(y) = logpdf(fx, y)
pred

@info "Init Loss = $(loss(y))"

ps = Flux.params(k)

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
