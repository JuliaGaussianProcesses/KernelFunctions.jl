@deprecate transform(k::Kernel, t::Transform) k ∘ t
@deprecate transform(k::TransformedKernel, t::Transform) k.kernel ∘ t ∘ k.transform
@deprecate transform(k::Kernel, ρ::Real) k ∘ ScaleTransform(ρ)
@deprecate transform(k::Kernel, ρ::AbstractVector) k ∘ ARDTransform(ρ)

# TODO: Remove deprecations in the constructors and docstrings of `GammaExponentialKernel`
# and `GammaRationalKernel`
Base.@deprecate_binding GammaRationalQuadraticKernel GammaRationalKernel
