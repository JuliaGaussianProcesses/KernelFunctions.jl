import .Flux.trainable

### Base Kernels

trainable(k::ConstantKernel) = (k.c,)

trainable(k::GammaExponentialKernel) = (k.γ,)

trainable(k::GammaRationalQuadraticKernel) = (k.α, k.γ)

trainable(k::MaternKernel) = (k.ν,)

trainable(k::LinearKernel) = (k.c,)

trainable(k::PolynomialKernel) = (k.d, k.c)

trainable(k::RationalQuadraticKernel) = (k.α,)

trainable(k::MahalanobisKernel) = (k.P,)

trainable(k::GaborKernel) = (k.kernel,)

#### Composite kernels

trainable(κ::KernelProduct) = κ.kernels

trainable(κ::KernelSum) = (κ.weights, κ.kernels) #To check

trainable(κ::ScaledKernel) = (κ.σ², κ.kernel)

trainable(κ::TransformedKernel) = (κ.transform, κ.kernel)

### Transforms

trainable(t::ARDTransform) = (t.v,)

trainable(t::ChainTransform) = t.transforms

trainable(t::FunctionTransform) = (t.f,)

trainable(t::LowRankTransform) = (t.proj,)

trainable(t::ScaleTransform) = (t.s,)
