using .Flux: trainable

Flux.trainable(::Kernel) = () # By default no parameters are returned
Flux.trainable(::Transform) = ()

### Base Kernels

Flux.trainable(k::ConstantKernel) = (k.c,)

Flux.trainable(k::GammaExponentialKernel) = (γ,)

Flux.trainable(k::GammaRationalQuadraticKernel) = (k.α, k.γ)

Flux.trainable(k::MaternKernel) = (k.ν,)

Flux.trainable(k::LinearKernel) = (k.c,)

Flux.trainable(k::PolynomialKernel) = (k.d, k.c)

Flux.trainable(k::RationalQuadraticKernel) = (k.α,)

#### Composite kernels

Flux.trainable(κ::KernelProduct) = k.kernels

Flux.trainable(κ::KernelSum) = (k.weights, k.kernels) #To check

Flux.trainable(k::ScaledKernel) = (k.σ, k.kernel)

Flux.trainable(κ::TransformedKernel) = (κ.transform, κ.kernel)

### Transforms

Flux.trainable(t::ARDTransform) = (t.v,)

Flux.trainable(t::ChainTransform) = t.transforms

Flux.trainable(t::FunctionTransform) = (t.f,)

Flux.trainable(t::LowRankTransform) = (t.proj,)

Flux.trainable(t::ScaleTransform) = (t.s,)
