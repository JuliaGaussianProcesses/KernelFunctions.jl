@deprecate transform(k::Kernel, t::Transform) k ∘ t
@deprecate transform(k::TransformedKernel, t::Transform) k.kernel ∘ t ∘ k.transform
@deprecate transform(k::Kernel, ρ::Real) k ∘ ScaleTransform(ρ)
@deprecate transform(k::Kernel, ρ::AbstractVector) k ∘ ARDTransform(ρ)

Base.@deprecate_binding GammaRationalQuadraticKernel GammaRationalKernel
